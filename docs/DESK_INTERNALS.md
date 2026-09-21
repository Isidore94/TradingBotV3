# Desk internals — the long form

Verbatim source text for the `Core loop / data flow` rules in
[`CLAUDE.md`](../CLAUDE.md), moved here on 2026-08-28.

**`CLAUDE.md` keeps the rule; this file keeps the reason.** That section had grown to
42 KB (~10,600 tokens) — 65% of a file that loads into *every* session — because each
rule carried the incident, the measurements and the trader conversation that produced
it. Those are worth keeping and are reproduced below unchanged. They are not worth
re-reading on every unrelated task.

**Read the matching entry here before you change the behaviour a rule governs**, and
whenever a rule's one-line form in `CLAUDE.md` is not enough to act safely. The rules
themselves are binding from `CLAUDE.md` alone — nothing here is optional context that
weakens them.

If you change a rule, change it in both places.

---

## The technical-integrity event log's layout (2026-08-31, snappiness packet 3)

`technical_integrity_events.jsonl` is the authority and has not changed: same
rows, same path, append-only, nothing removed or rewritten. It measured **618 MB**
on 2026-08-31 with no retention, and the after-close wrap-up replayed it every
evening by streaming and `json.loads`-ing every line to keep the `level_resolved`
rows - a small subset. That is an hour-class job, and although it runs on a
background thread the GIL means an hour of hot parsing steals GUI-thread time all
evening.

**Beside it now sits `technical_integrity_events_resolved.jsonl`, a DERIVED
sidecar.** `_append_event` writes the main log FIRST and then mirrors any
`level_resolved` row; that second append is swallowed on failure, because losing
a derived line costs a catch-up scan and nothing else.

**The watermark rides on the rows, not in the header.** Every sidecar line carries
the main log's byte offset at the moment the row it mirrors was appended, so the
last line IS the watermark and the file stays append-only - a header watermark
would have to be rewritten on every event. `sync_resolved_sidecar` therefore has
three honest outcomes: **current** (offset == size, no work), **caught up**
(stream only the tail past the offset), and **rebuilt** (no sidecar, a torn line,
or an offset past the end of the log, which means the log was replaced under it).
Both the build and every catch-up end by recording how far they actually read,
because the last resolved row is rarely the last LINE and without that every later
sync re-streams the same tail forever.

`load_resolved_technical_integrity_events` prefers the sidecar and falls back to
the full stream on any doubt; `use_sidecar=False` forces the old path, and a test
asserts the two return the same rows in the same order.

**The month roll is deliberately NOT built.** Renaming the live log into
`-YYYYMM` segments needs every reader to see the live file plus the segments, and
`research_warehouse/ingest_existing.py` registers the log as a `BronzeArtifact`
whose `resolve_path()` returns exactly ONE path. Teaching it several is a change
to the warehouse's bronze contract - a locked area with its own decision log -
so shipping the roll would have left the warehouse silently ingesting one month.
The sidecar removes the replay cost, which was the GUI-freeze problem; the roll
would only bound disk growth.


- Entry: `launch_gui.py` → `scripts/ui/app.py` (PySide6 Trading Desk). There is one desk role and no flag to change it — the Desk Link satellite role was retired 2026-08-08 and its code was **removed 2026-08-24** (P1.5): no `desk_link` package, no `ui/satellite.py`, no `--satellite`/`--desk-role` flags, no Settings ▸ Desk Link tab. `scripts/gui.py --ui tk` is the legacy Tk UI kept during migration.
- Market data: IBKR TWS/Gateway on `127.0.0.1:7496` (`ibapi`) primary, `yfinance` fallback; bar source is tracked per scan. See `docs/BROKER_ADAPTERS.md`.
- Engines: `scripts/master_avwap.py` (+`master_avwap_lib/`) D1 AVWAP swing scanner; `scripts/bounce_bot.py` (+`bounce_bot_lib/`) intraday M5 bounce detector; `market_prep/` pre-session services.
- Inputs: plain-text watchlists (`longs.txt`, `shorts.txt`, `swinglongs.txt`, `shortswings.txt`) in the user-selected shared "home folder".
- Mutable state lives in that home folder — `C:\TradingBotData`, a plain LOCAL folder on the desk SSD. **There is no cloud drive: Google Drive/OneDrive were removed 2026-08-10 (decision 0015) and are no part of this system.** It holds compact operational state: watchlists, reports, JSONL/CSV evidence logs. Per-machine caches + diagnostics live under `%LOCALAPPDATA%\TradingBotV3` (`scripts/project_paths.py`).
- Storage tiers: desk SSD is local/staging; the **DAS file server `\\MINI-PC\Trading Bot Data` is the durable tier** (expandable to ~100TB) and holds the research lake, the AI store, and cold subtrees pushed hourly by `C:\TradingBotData\_tools\push_cold_to_das.ps1`. Write local first, move to the DAS after, so a file-server outage costs throughput and never correctness.
- Research warehouse (Phases 0–8 implemented; plan.md Phase 3 owns live evidence and post-slice work): very large research files (bar archives, feature/outcome Parquet) go to the DAS research lake at `research_store_dir` (`local_settings.json`; env `TRADINGBOTV3_RESEARCH_DIR`), configured 2026-08-10 to `\\MINI-PC\Trading Bot Data\research_lake` with a machine-local spool at `%LOCALAPPDATA%\TradingBotV3\research_spool` — a separate append-only storage class (decision 0014) that is NEVER inside the `C:\TradingBotData` home folder (`scripts/research_warehouse/config.py` refuses such paths; unset = warehouse fully disabled). The refusal now rests on storage-class separation and cold-push scope, not sync quota (decision 0015). Locked contract: `docs/ULTIMATE_SETUP_DATABASE_PLAN.md`. Builder-level implementation decisions are logged in `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`; dataset keys/identities in `docs/RESEARCH_WAREHOUSE_ERD.md`. Shadow-only additive evidence — zero detector/score/alert influence. **The build runs INSIDE the desk process, so its reads are session-scoped** (2026-08-27): partitions are MONTH-keyed, and `read_table(partition).to_pylist()` in three steps materialised 8.7M rows / **15.4 GB** to use one 588k-row session, which is what made the desk jump to 8-13 GB after every swing-scan slot and fall back minutes later. Narrow through `ResearchStore.read_rows` (Arrow-side `symbols` / `interval_start_range`), never by filtering a materialised list - and never widen `_run_outcomes` to a date filter, because its walk runs FORWARD across sessions (BD-66/BD-69/BD-74). A SNAPSHOT over 64 MB is stored whole but NOT `json.loads`-ed, and the UNCHANGED watermark check is answered from a chunked hash before any `read_bytes` - `master_avwap_setup_tracker.json` is 1.03 GB and was being read whole just to be discarded (BD-73). The growth is month-keyed, so it worsens all month and resets on the 1st; check the calendar before treating a new report as new.
- Shadow engines `scripts/market_state.py` (via `market_state_bridge`) and `greatness_monitor` (via `greatness_shadow`) run beside the legacy champions and emit JSONL promotion evidence only.
- Review-learning loop: Alert Center decisions → `alert_review_events.jsonl` → `review_learning.py` scoreboard → AI-curated `review_policy.json` → chart annotations (queue ordering gated to FIFO). See `docs/REVIEW_LEARNING_LOOP.md`.
- **Alert Center review pane layout** (trader rule 2026-08-20): the charts own the pane, and between them and the tab strip there is at most ONE slim row — the verb row, which advances the review queue and must never cost a click. **The arm bar stays UNDER the chart** (same day, second pass, `4c05de5` - the trader: "I also need my m5 and D1 alert hotbuttons back on the bottom of the visual chart... I also need the ability to input a ticker manually as well"); only the CaptureRail moved, to the **Capture** tab. `AlertChartReview(dock_arm_bar=..., dock_capture_rail=...)` makes placement a HOST decision (`AlertCenterPanel` passes `dock_arm_bar=True, dock_capture_rail=False`), and hosts with room (`SymbolSnapshotDialog`, Chart Review workspace) keep the docked stack. This line said "Armed tab" until 2026-08-26 - the first pass - and the 2026-08-25 GUI proposal built a wrong recommendation on it; never propose moving the arm bar without asking. The rail's five-second/no-mouse contract moved with it: a `QShortcut` bound inside a hidden tab page never fires, so `AlertCenterPanel` rebinds `CaptureRail.action_shortcuts()` at panel scope and the rail binds none of its own there (two live bindings for one sequence is an ambiguous shortcut and Qt fires NEITHER). Armed state stays visible via the tab count and `armed_summary` on the verb row.
- **The capture rail is a recorder that now moves the queue** (trader rules 2026-08-20): a VETO and a LIKE each retire the chart the way "Not today" does; a NOTE never does, because it is written ABOUT the chart in front of you. "Veto D1 - but M5 today" writes an ordinary veto row and emits a REQUEST - `AlertCenterPanel` performs the M5 Focus placement, so the Focus store keeps one writer; place first, retire second, and a failed placement still retires. Known limitation, deliberately not papered over: the veto cohort therefore counts a day-traded name as vetoed. The hypothetical-stop CONTROL was removed; `EVENT_HYPO_STOP` stays in the annotation schema so existing rows remain readable. **What the LIKE offers is `MAIN_CLAIM_GROUP` + `EXTRA_CLAIM_IDS`** in `ui/annotations/setup_claims.py` (definition moved out from behind Qt 2026-08-24 so headless `ai_summary` derives its picklist caveat from the same source via `offered_setup_claims()`; the rail re-exports both, so old imports keep working): all of Main swing, then the three post-earnings families and `second_dev_breakout` (trader, 2026-08-21). Keys run `1234567890qwerty...` in list order, so the nine main-swing digits never move; a row's label starts with its own key, which keeps QListWidget type-search agreeing with the shortcut. Admitting another family is one line there, never a migration.
- **The nightly journal slot speaks, and the Questrade chain is watched** (AI-layer review packets, 2026-08-24 — `docs/archive/analysis/AI_LAYER_REVIEW_2026-08-24.md`): `run_journal_backfill` returns `failures` beside `status` and the nightly ledger reason names the first three failures and PRINTS the count it dropped, plus only what the night measured — a skipped reconcile says "skipped", never "0 mismatches". Reconcile MISMATCHES never marked the slot failed (refuted premise, regression-pinned); only exceptions do. `scripts/journal_health.py` surfaces a dead Questrade OAuth refresh chain in Journal ▸ Health and System Health; repair is a TRADER action — paste a fresh refresh token into Journal ▸ Health ▸ "Questrade refresh token" (key `journal_questrade_refresh_token` in `%LOCALAPPDATA%\TradingBotV3\local_settings.json`, a secret-bearing file; single-use rotating chain, weekly paste is the routine, never the env var). **The chain has ONE owner** (2026-08-25): Questrade rotates on every refresh — a success invalidates the access token it replaces AND consumes the refresh token it was given — so "Pull today now", the gap backfill and the nightly slot were three consumers of one single-use chain and broke it eleven minutes after a paste (import OK 20:54:59, backfill 20:59, `400` on the refresh endpoint 21:06:51). `refresh_access_token` now holds `local_writer_lock`, **re-reads the token inside the lock** (a caller that waited spends what the winner LEFT), and writes the four rotated values through `project_paths.save_local_settings()` in one temp-file+`os.replace` save; `_authorized_get` answers a 401 caused by someone else's rotation by picking up THEIR access token instead of burning a refresh. A failed refresh saves nothing and never clears the stored token. **The attempt cap counts failures against a DAY, not a cause**, so days that failed while the chain was dead were skipped forever — `self_heal(include_exhausted=True)`, passed only by "Retry failed Questrade days", lifts it for one deliberate run while the nightly keeps it; `attempts` is never rewritten and the run reports `reopened_exhausted`. **Not every FAILED day is repairable:** 44 of 45 `activities report trades…` days predate 2026-06-10, the executions endpoint's retention horizon, so retrying them can never work — recovering them from `/activities` or labelling them permanently uncovered is a trader decision needing a new coverage status. Lesson paid for twice: address home-folder stores by their `project_paths` named constants — Focus Pick Review resolved CSVs by name under `PERSISTENT_DATA_DIR` while they live in `data/runtime` and shipped a blank page from 08-18 to 08-24, and its fixture encoded the same wrong assumption.
- The day-trade pass (trader-directed, 2026-08-31). In as many words: *"many times I really like this stock for a daytrade but it has this ONE issue"* and they pass, and that judgement was going nowhere. The trader asked for a section **under the existing Note area** of the capture window with tickable reasons, several allowed at once, alongside the free-text note; and: *"if the M5 data for the symbol is already in memory at that moment, attach it, so an AI can later see the chart as it was"*, with the explicit fallback *"if that is hard, just store the exact timestamp and the AI can read the charts by it."* Five things follow and are binding.

  **A pass is not a veto.** A veto says the chart in front of the trader is not for today; a pass says the day trade WAS there and one thing stopped them. Separate `event_type` (`pass`), separate vocabulary FAMILY (`pass_reasons_v*.json` beside `veto_reasons_v*.json`, loaded by `load_pass_vocabulary`), and no shared codes. Folding the five new reasons into the veto list would have restamped `vocab_version` on every veto cohort already accruing forward returns, for two lists that answer different questions. `ui/annotations/vocabulary.py` was generalised to load any family and validates `vocabulary_id` against the filename, so a file that declares the wrong family fails closed rather than writing codes under the wrong identity.

  **It never retires the chart.** CLAUDE.md's rule is that a veto and a like each retire and a note never does; a pass sits on the note side, because it is written ABOUT the chart the trader is still reading. `AlertChartReview._on_captured` and `SymbolSnapshotDialog._on_captured` both key on `EVENT_VETO` / `EVENT_LIKE_CLAIM` only, so this is structural rather than a new branch. Pass-to-retire was not asked for and is not built.

  **Several reasons per pass, written in VOCABULARY order.** `_clean_pass_codes` dedupes and reorders, because two passes citing the same two reasons have to compare equal months from now and click order carries no meaning worth preserving over that. The starting five are the trader's own labels, unedited: Poor market conditions / Low rvol / LRSI/SMI incongruency / Incoming Horizontal / Other incoming S/R.

  **The bars are a SIDECAR, and they are never fetched.** One session of M5 bars is ~78 RTH bars and well over the store's 4096-byte single-write cap, and raising that cap would trade the annotation stream's confinement property (a torn tail costs exactly one row) for a convenience. So `ui/annotations/pass_bars.py` writes `trader_annotation_bars/<event_id>.json` FIRST and the row references it second — a sidecar with no row is a few orphaned KB, a row pointing at nothing would be a lie in a permanent record. The bars come from a host-supplied provider (`CaptureRail.set_m5_bars_provider` ← `SymbolSnapshotWidget.cached_m5_bars`), which copies what the pane already DREW; the rail reaches for no bot, service or feed, and a provider that raises costs the attachment and never the row. Only the newest session is kept: the desk hands out two because an ATR(14) needs warm-up bars, and a pass is about today.

  **Nothing in the chain reads back.** The pass rows are analysis-only evidence like the rest of the stream: no mute, no suppression field, no score, no gate, no alert. Deliberately NOT changed, and DECIDED rather than pending (trader, 2026-08-31): `pick_feedback._ANNOTATION_DECISIONS` still lists `veto`/`like_claim`/`note`, so a pass does NOT mark a symbol "Reviewed today" — *"that flag feeds the scanner report and several badges. Making a pass count as reviewed touches scanner-side code, so it should be its own small job if you want it."* A test pins it. The same conversation closed the other question: a pass never retires the chart and needs no option for it — *"if you pass AND want the chart gone, just hit veto after. You get both behaviors without a new rule."*
- **Veto vocabulary is versioned, and codes are never reused** (`scripts/ui/annotations/vocabularies/`): v2 shipped 2026-08-20 replacing the "S/R cluttered" slot with "Compressed" as a NEW code; **v3 shipped 2026-08-21 adding "SMA incoming"** (hotkey `0` — 1-9 were spoken for and renumbering learned digits costs more than an out-of-run key) and changing nothing else. Every older version stays on disk and stays loadable. Cohort identity on the way IN is still `(vocab_version, reason_code)` — `veto_cohort_source(code, version)` — and rows are never rewritten. What changed is the way OUT: `canonical_veto_cohort(source)` pools sources whose reason DEFINITION (code, label, hint, note rule) is identical across versions, so an additive bump no longer restarts a reason's forward record, and the eight v1↔v2 cohorts the earlier bump split are pooled again. It is applied only when the performance rollup is rebuilt (`_rebuild_pooled_performance`), never at write time — pooling on the way in would destroy the distinction permanently. A reason introduced later (`compressed` in v2, `sma_incoming` in v3) keeps its own cohort. Never assert a literal `vocab_version` in a test; assert against the loaded vocabulary.
- **D1 charts carry a volume underlay and an earnings ribbon** (`candle_chart.py`): both are drawn INSIDE the price view (volume in the bottom 18%, earnings on a reserved top rail) rather than as stacked sub-plots, because this column has no height to spare. Neither votes on the price range. The earnings ribbon's headroom is reserved for EVERY symbol so two names at the same price never draw at different scales. **The earnings cache holds no future dates for any symbol**, so the next report is projected from median cadence (`scripts/earnings_projection.py`) and labelled `est` everywhere. Payloads are built on the ChartDataService worker beside the levels - never on the paint path.
- **Nothing expensive belongs on the Qt thread, and "expensive" includes a stylesheet** (fluidity pass, 2026-08-21). Measured over 3h20m of live use: **1843 stalls >50 ms, 1008 s blocked**, plus the two GC freezes. The trader's suspicion - the DAS - was ruled out with numbers: every hot path resolves to `C:\TradingBotData` or `%LOCALAPPDATA%`, the GUI holds no reference to the research store outside two worker-thread tiles, `\\MINI-PC\Trading Bot Data` was momentarily unreachable during the measurement and a miss on it cost **0.0 ms** (it resolved again the same afternoon - the share drops and re-establishes, which matters for the overnight AI-store and warehouse writes but never for the GUI). What it actually was: (a) **per-widget `setStyleSheet`** - Qt parses CSS and re-polishes per widget, and both busy lists rebuilt themselves whole (105 focus chips, up to 250 feed rows), which is also where the cyclic garbage came from; (b) **uncached file reads** - `project_paths._load_local_settings` had ~100 call sites and re-parsed every time, `review_events.load_review_events` re-parsed 5.8 MB per call; (c) **`BarSeries.as_bar_dicts` on Qt** for every armed and Focus symbol per poll, against its own docstring. Rules now: lists **diff, never rebuild** (`FocusSideEditor.refresh`, `FocusStatusChip.update_state`); widget variants live in `theme.qss` keyed on object names and dynamic properties, with pre-mixed rgba in `theme._derived_tokens`, so a variant costs a property set and not a parse; materialization goes through `ChartDataService.cached_bar_dicts`. Also: **the theme sizes fonts in px**, so `QFont.pointSizeF()` is **-1** and arithmetic on it is a bug - `setup_delegate._resized` is the one place that scales a font, and it stays in whatever unit the font uses. `ui.app.install_qt_message_rate_limit` prints each distinct Qt message once and counts the rest, so a warning storm from inside `paint()` can never again cost a frame.
- **A burst of one signal is ONE reaction, and the coalescing lives at the LISTENER** (2026-08-31 desk lockup). 07:37-07:53 that morning: ~500 s of GUI-thread blockage in a 16-minute session, 216 s blocked in the 5.5 minutes after 07:45, ~80% frozen between 07:50 and 07:52, single stalls of **44.3 s**, 15.9 s and 15.2 s, Windows reporting Not Responding, and the trader killing the desk twice - each restart re-running the 07:30 swing scan. Memory was fine (~2 GB WS); this was not the 2026-08-27 warehouse bug. The cause: at 07:41:58-07:42:11 the Alert Center drain adopted **45 staged picks into M5 Focus one at a time**, ~300 ms apart, and `FocusPickStore.add()` notifies per add - correctly, several surfaces need it - but **five listeners each treated one add as a full rebuild**: four editor rebuilds plus a `pick_feedback` read plus a forced snapshot WRITE (Focus board); both alert feeds destroyed and reconstructed, up to 350 widget trees each with its own stylesheet (`_rebuild_feed`); a full setups-viewport repaint through `SetupTableDelegate` (the hottest stack in the log, ~300 samples across paint lines 78-152); the strength board rebuilt as HTML and re-parsed by `setHtml`; the price-alert combo cleared and refilled. The ~300 ms spacing between adoptions WAS that work. So: the signal contract is untouched and `ui.timer_utils.SignalCoalescer` sits at each listener - a **200 ms leading-edge window with a trailing fire**, where later requests fold in and deliberately do **not** restart the window. A synchronous drain loop therefore lands whole inside one window (one reaction), while a plain restart-on-signal debounce would be starved by a stream arriving faster than its window. 200 ms is the trader's ceiling, not a target. Two more defects went with it: `FocusSideEditor.refresh()` **documented itself as a diff and still emptied the flow layout and re-added every chip on every call** (90 layout operations on a 45-name board to change nothing) - the unchanged case is now zero layout work and `FlowLayout` grew `insertWidget`, because `QLayout` has no generic insert and its absence is why the teardown existed; and `record_bounce_alert` rebuilt four editors and re-read the feedback file to light ONE chip's badge, so it now touches only the matching chip (`_bounce_state` is still written first, so a name joining Focus after its alert still gets the badge). The DESK drain additionally adopts at most `AUTO_ADOPT_BATCH_LIMIT` (10) staged picks per 30-second cycle, trader-approved 2026-08-31 (*"cap the auto-adopt batch and slow the redraws"*): **pacing, never policy** - the freshness gate, the flip barrier, ownership markers and AWAY/EVENING's refusal are all upstream and untouched, the cap counts adoptions rather than iterations, a deferred pick is **not** marked seen, and **no pick is ever dropped** (a cap that withheld one would be the suppression field this chain deliberately does not have). The GUI-thread GC controller was deliberately left alone: its ~600 ms young sweeps were a symptom of this churn, and its delay-never-cancel and GUI-thread-only invariants are load-bearing.
- **All cyclic GC runs on the GUI thread, and activity may DELAY a sweep but never CANCEL one** (`install_gui_thread_gc` / `_GuiGcController` in `ui/app.py`). `gc.disable()` is in force process-wide so Qt wrapper destructors stay on the owning thread — which makes that timer the process's ONLY collector. `d0aebd5` gated both sweeps on input idleness with no upper bound, and a trader working continuously produces input every few hundred ms, so nothing was collected while the desk was in use: on 2026-08-21 it reached **8 GB in 90 minutes** and froze for **298 s and then 200 s** in the sweeps that finally ran, recovering to 1.9 GB. Every wait now carries a deadline in ticks (`young_deadline_ticks=5` ≈ 10 s, `full_deadline_ticks=90` ≈ 3 min at the 2 s tick). Any future "wait for quiet" here needs a bound, for the same reason.

  **The startup heap is frozen out of every later sweep (2026-08-31, snappiness packet 2).** Because that timer is the only collector, every sweep's cost is a GUI freeze, and the 2026-08-31 stall log put **6.5 of the day's ~78 minutes of freeze inside the collector** - gen-0 sweeps averaging ~300 ms and full sweeps ~770 ms. Most of what those sweeps walked can never be garbage: the widget tree, the theme, every imported module. So `main()` now runs one `gc.collect(2)` and then `gc.freeze()` immediately after `MainWindow` is shown and before the stall watchdog installs. The ORDER is the rule - collect first so only survivors become permanent; freezing first would make every piece of startup garbage immortal instead of collecting it. Nothing about `_GuiGcController` moved: same cadence, same 250 ms idle wait, same every-30th-tick full sweep, same bounded deadlines. The same sweeps, over a smaller heap.
- **A candle's four prices carry an invariant the chart used to assume** (`scripts/ui/bar_integrity.py`, 2026-08-21): `low <= open, close <= high`. The y-range is built from lows and highs while the BODY is drawn from opens and closes, so a bar that breaks it paints a solid column through the whole viewport while the axis still reads normally — that is the "massive green candle" a corrupt row produces. `CandleItem` now draws such a bar **dashed, hollow, in the caution role with its body clamped into its own range**, `price_range()` keeps it out of the scale, and a bottom-left note says how many bars that happened to. Never silently dropped: missing data is uncertainty, never confirmation. `ChartDataService` logs each one once to `%LOCALAPPDATA%\TradingBotV3\diagnostics\bad_bars.jsonl` with symbol/timestamp/OHLC/provenance. `range_outliers()` additionally OBSERVES well-formed bars whose range dwarfs their series (what a daily row dropped into an M5 cache looks like) — logged, never redrawn.
- **"Holding highs" is measured in ATR and it expires** (`scripts/regime_pause_hold.py` + `scripts/indicators/atr.py`, trader rules 2026-08-21). The regime-pause watch captioned MRK "holding highs" while its high of day was 75 minutes old and price was fading; the batch label is applied to every symbol in the sweep, and the qualifying predicate's third branch (`window_excess`) admits a name that is merely falling less than SPY. Two rules came out of it. **Distance is in ATR, never percent** - M5 ATR ran 0.084%-1.160% of price across ONE batch that day (14x), so a fixed percentage is simultaneously too loose and too tight; tolerance is **1.0 ATR** and the extreme is taken from the last completed bar's session while the ATR may use earlier sessions for warm-up. **A row is good for 15 minutes** from the later of the alert and the last new extreme - a new HOD/LOD refreshes the clock - and is then **deleted from the review queue only**: History, `alert_review_events.jsonl` (a `hold_expired` row) and the tracker's outcome rows keep it, so the rule stays gradeable. Uncertainty never deletes: no bars, no ATR, no readable stamp all mean KEEP. Shorts mirror throughout. **The detector-side gate is BUILT** (2026-08-21, golden fixture `regime_pause_sweep_v1` frozen first per plan.md sec 5): `_sweep_regime_pause_bangers` now requires `regime_pause_hold.hold_state(...).holding` IN ADDITION to the existing `still_trending or made_new_extreme or window_excess` defiance test, so the flagged set can only shrink. It is handed the FULL cached series, not `sym_today`, because an ATR(14) needs fifteen bars and the sweep fires when there are nine; `hold_state` takes its ATR from everything supplied and its extreme from the last completed bar's session. **Being AT the extreme needs no ATR** and is holding regardless - without that carve-out the gate silently switches the detector off early in a session, which is what three champion tests caught. The feed line now carries a per-symbol measure (`HTFL (new HOD), MRK (0.7 ATR)`) instead of one batch phrase, and `ui/models/bounce.py` parses it into the per-row caption; a bare symbol still reads as the old phrase. **A second gate landed the same day** (same fixture, frozen and re-frozen again): a flagged name must ALSO have broken the previous session's high (longs) or low (shorts) and be on the right side of session VWAP. That pair is the M5 Focus adoption gate (trader rule 2026-08-14) and is CALLED, not restated - `passes_focus_adoption_gate` - with the numbers read off the cached M5 series by `regime_pause_hold.session_levels` (prior session extremes from the series itself, session VWAP from `chart_snapshot.session_vwap_series` and never BounceBot's dynamic/EOD VWAP). UNKNOWN fails here as everywhere: no prior session in the cache, or a series with no volume, means no alert rather than a free pass. On the day's real batch the two gates together take 38 longs to 18 and 29 shorts to 18 - MRK and GFS among the drops; the VWAP half bound on nothing that day, which is expected since a name near its high is nearly always above its VWAP. **With-trend rows auto-join M5 Focus** (trader rule 2026-08-27, after reviewing 21 such charts in nine minutes and hand-adding twelve): a LONG holding highs on a bullish day or a SHORT pressing lows on a bearish day is placed on M5 Focus by `AlertCenterPanel._auto_focus_regime_pause` (DESK only; store write + auto-pick marker, never `FocusService.add`; day label = `resolve_discovery_env` over the live env and the opening read; `regime_pause_auto_focus` evidence row) and skips the review chart. Rule in `scripts/regime_pause_focus.py`, pure. Counter-trend rows, a non-directional day, the trader's own Focus entries and every failure path keep the old behaviour - the row charts.
- **The overnight runner has a fourth, deterministic slot** (`veto_cohort_grading`, `scripts/ai_jobs/cohorts.py`, added 2026-08-20): it grades the veto cohort forward and calls no model. `default_slots()`' rule is "later phases append; they never reorder these" - follow it. Sideless picks are counted and named, never graded (a blank side reads as LONG in `human_focus_tracking`). The `trader_judgement` evidence scope exists but is **deliberately absent from `DEFAULT_SCOPES`/`TICKER_BRIEF_SCOPES`**; exercise it with `run_ai_jobs.py --scopes trader_judgement`. Nothing in this chain may reach a detector, score, alert, watchlist, Focus, the review queue or `review_policy.json`.
- **Long vs short survives a statement, and layering later exports must not double the year** (trader direction 2026-08-28: *"lets add a function to be able to take these files, and new ones throughout the year that layer on top"*). Two defects the first build carried, both found by measurement rather than review. **(1) The uid was positional.** It hashed the file's row index, so a January-to-December export - which lists the same January trades at different positions - made **884 of 884** real trades look new; a one-row shift was enough. Identity is now `fill_signature` (account, date, symbol, side, quantity, price, commission, currency) plus an ordinal counted *within that signature*, so it is identical in every export containing the trade. Proven on the trader's own two files: all 884 of the 2026 file recognised inside the 2025-26 file, and re-importing either in any order leaves 1,516 executions and 202 cash rows unchanged. **(2) Direction was a coin flip.** A statement has no clock, and the file lists a same-day round trip SELL-first **227 times out of 227** - which makes row order a SORT, not a sequence. The assembler breaks a timestamp tie on the execution uid, a hash, so **86 of 199** same-day trades came out SHORT at random. The fix is that Questrade *says so in the Description*: `"... COMMON STOCK SHORT."` on a short sale and `"... COVER SHORT."` on the buy that closes one. `leg_rank` ranks each row by what it does to the position (opens before closes) and that rank is the uid's sort prefix, which is where the intra-day order has to live because every row on a date shares midnight. It resolved all 227 - **169 long, 58 short** - and every one of the 58 carried BOTH markings, so the halves corroborate each other rather than being read off one row. Absence of a marking is itself the answer, because Questrade marks every short. **What it still cannot split**: a day holding both a marked short and an unmarked long in the SAME symbol (3 days of 439 on the trader's history - IOVA 2025-07-23, QBTS 2025-07-17, ASTS 2025-05-19). The assembler groups a symbol into one position, so it blends what were really two trades; the day's money stays exact because everything closed, and `reconcile_statement` names those days rather than resolving them silently. **`reconcile_statement` is the trader's own proof** and writes nothing: it adds the file up by hand - for a symbol whose quantities net to zero across the file, the sum of its Net Amount column IS the realised P&L - and compares that to what `rebuild_trades` assembled. The two share only the file, so a disagreement is an assembly defect and the per-symbol rows say which symbol. Symbols still holding a position are **excluded, not zeroed**: cash has left with no realised P&L against it. Measured across both files: statement **$5,298.81** vs journal **$5,299.05**, difference **-$0.2386** over 428 closed symbols, every symbol inside two cents, and commission **$713.68 both ways**. What it does NOT prove is the parse itself - both sides read the same parse - which is why it is a demonstration and only the trader's Questrade year-end numbers can close it. Importing the 2025 file also dropped NEEDS_REVIEW trades from **23 to 5** by giving carried-in positions their real opening fills.
- **The tax number is the broker's, never ours** (trader decision 2026-08-28: *"Statement is source of truth for final pnl/tax purposes"*, which is a stronger rule than the day-level authority above and needed its own answer). Everywhere else the journal RECOMPUTES a trade's P&L - average-cost matching, price x quantity, pro-rated costs - because that is what makes per-trade attribution, R multiples and per-setup statistics possible at all. It is also, unavoidably, arithmetic of our own: Questrade books each row's Gross Amount rounded to the cent while the assembler multiplies at full precision, so the two drift by **-$0.2386 on $5,298.81 of realised P&L across 428 closed symbols**. Immaterial for deciding what to trade; not the number to put on a return. `scripts/journal_tax_report.py` therefore recomputes nothing. For every fill it takes `raw_executions.net_amount` - the broker's own statement of what that fill did to cash - and adds them up; **for a FLAT position that sum IS the realised P&L**, because every share bought was sold, so no cost-basis model is needed and none is used. That required one normalization first: Questrade and Flex already state that figure in the trade's own currency but the IBKR transaction file states it in the account's BASE currency, so the IB importer now divides by the row's implied rate before storing and keeps the base figure in `raw_json` - `net_amount` means one thing for every broker in the store or the sum is meaningless. **What it refuses to report is the point.** A position that is not flat contributes nothing (cash has left with no realised P&L against it; including it would report an open trade as a loss). A position with an invented opening fill contributes nothing - a `SYNTHETIC_OPEN` leg means the proceeds are real and the cost basis is not, and these are listed BY SYMBOL so the trader knows which file fixes them; on the real data that count went from 23 to 5 when the 2025 export arrived. A fill carrying no broker-stated amount disqualifies its whole position, because the IBKR socket path records none and mixing a stated figure with a recomputed one produces a total that is neither. Nothing is estimated to fill a gap: a tax figure that quietly interpolates is worse than one that names the symbol it cannot answer for. A `VOID_EXECUTION` row - retired by a correction or by the file-authority rule - never reaches a total, because it no longer describes the account. CAD is the tax currency and converts **per fill at the Bank of Canada rate booked for that fill's date**, never one rate for the year and never a broker's internal rate; a fill whose date has no booked rate leaves its position's CAD total `None` and the count is reported. Accounts are reported separately with their tax status, and currencies are never added together (I6). A position spanning the year end is reported whole rather than cut in half, because splitting it would invent a cost basis for one half and proceeds for the other. `cross_check_against_journal` puts the recomputed figure beside the stated one per account - two independent routes to one number, never blended - and on the trader's own two brokers it reads: broker **$8,219.81**, journal **$8,220.05**, difference **-$0.2385**, which is precisely the known Questrade cent-rounding (IBKR reconciles exactly). Reached from Journal > Fees > "Realised P&L for tax...", with a year picker and a CSV listing every counted position and every excluded one with its reason.
- **A broker file outranks the live sync on MONEY, and never on time** (trader decision 2026-08-28, from *"these should be sources of truth moreso than the auto input IMO"*). The blunt reading was put to the trader with its cost measured, and they chose the split: **neither Questrade's nor IBKR's downloadable file carries a time of day**, so letting a file take over every day it covers would discard the only intraday timestamps the journal owns - every session bucket, every "what time do I trade best" question, and the `journal_trade_shape` entry-time tags built on them. So the sync keeps a day when the two AGREE, because it alone knows when each fill happened and how it was split; the file takes a day when they do NOT, because it is the broker's own statement of the money and that is the number a tax return uses. **Agreement is measured in cash, per `(account, day)`, not per trade** - a trade can span days so a day's P&L is not even defined, while a day's cash impact is: `sign x quantity x price x multiplier - commission - fees`. That is COMPUTED rather than read off the file's own Gross/Net column, deliberately: Questrade reports those in the trade's currency and IBKR in the account's base currency, so the two columns are not comparable to each other while this formula is. **The tolerance is per fill, not flat** (`TOLERANCE_BASE` + `TOLERANCE_PER_FILL` x fills) because Questrade books each row rounded to the cent while the journal recomputes price x quantity, so a busy day accumulates fractions of a cent per fill - a flat threshold would fire on rounding or miss a real difference on a quiet day; the per-fill cent is a bound, not a guess, since the worst measured single execution differed by half a cent. **Taking a day over is APPEND-ONLY**: invariant I3 forbids deleting or editing a broker row, so the sync's executions are retired with `VOID_EXECUTION` adjustments carrying the day, both cash figures and the difference in their stated reason. They stay in `raw_executions` and in the audit list, stop applying at the next rebuild, and a superseding record undoes the whole thing - which matters because the trader can change their mind about a day. **A day the file does not mention is a gap, not a disagreement**, and is never touched: taking it over would delete real fills for no reason. Proven end to end against the trader's own 2025-26 Questrade export with a simulated live sync over August, one day of which was deliberately given only half its fills: **18 shared days, 17 agreed and kept their real 09:45 timestamps, and the crippled day was taken over on a $3,116.49 difference** (3 rows voided, 5 written); 15 August trades still carry a real entry time afterwards. The same comparison runs as a **dry run** behind "Check a statement...", so the trader can see which days would move before any of them do.
- **IBKR's transaction file, and the commission sign that was costing money** (trader direction 2026-08-28: *"we need IB integration as well... we would want to manually input a file as well"*). `scripts/journal_ib_transactions.py` is a SEPARATE reader from the Questrade one, because three things differ and each silently produces a plausible wrong number. **(1) It is a SECTIONED csv** - every line names its section (`Statement` / `Transaction History` / `Summary`) and says `Header` or `Data`, so the header must be tracked PER SECTION; a plain `csv.DictReader` reads the first table's header and misaligns every row after it. **(2) Money is in the BASE currency and prices are not.** On the trader's file `Price` is USD while `Gross Amount` and `Net Amount` are CAD - a 3-share sell at 366.19 USD books a gross of 1516.905456 - so passing both through computes a USD gross and subtracts a CAD commission from it. Executions are stored in the trade's OWN currency and the cost is converted by the rate the row itself implies, `|Gross| / |qty x price x multiplier|`. That rate is IB's own for that trade; it is recorded in `raw_json` as evidence and **deliberately never booked into `fx_rates`**, which is a Bank-of-Canada table by design - a broker's internal rate is not the rate a tax return uses. Across 608 rows the implied rate ran **1.35530-1.45270**, the USD/CAD band for the period, which is the check that the reading is right rather than a coincidence. The option **multiplier is inside that denominator**: without it the implied rate comes out a hundred times too large and corrupts the commission with it. **(3) Account numbers arrive MASKED** (`U***2524`, `U***7396`). A mask cannot be an identity - the same account reached through Flex carries its full number, and treating the two as different accounts splits one position in half - so `resolve_account_number` unmasks against accounts the journal already knows and only when **exactly one** fits; the filename is another candidate, never an override, because an IBKR export is named for one account but can contain rows for several. An unresolved mask keeps its masked form and is REPORTED. An `Assignment` is a real fill (`Buy 100 ROUNDHILL MEMORY ETF (Assignment)`) - dropping it leaves the position open forever with nothing that can close it. Everything the Questrade reader learned still holds: no time of day, midnight market-local, and a file never writes into a `(broker, account, day)` a richer source covers. **THE COMMISSION SIGN.** `upsert_executions` and the assembly path used to `abs()` commission and fees. Every importer already normalizes a charge to a positive cost, so removing it is a no-op for Questrade, Flex, the socket, CSV and manual rows - but **18 of 609** IBKR fills carry a commission CREDIT, and `abs()` turned each rebate into a charge, overstating the year's cost by **twice** the credit. Measured: that single sign was the ENTIRE $2.17 by which the IB file and the journal disagreed. With the importer owning the sign, IB reconciles to **-0.0000 across 150 closed symbols** and commission matches to four decimals - exact, where Questrade is off by cents, because IB writes full-precision amounts while Questrade rounds each row. Questrade's own reconciliation was re-measured after the change and is unmoved. One "Import statement file..." button serves both brokers and **reads the broker from the file's contents**, never from its name: both ship `.csv` and the name is whatever the trader saved it as, so asking them to pick would be asking them to get it right every time.
- **A broker statement is authoritative for money and blind to time** (trader-supplied file, 2026-08-28: *"i can easily get us yearly reports from questrade so long as we can process these files"*). Questrade's executions endpoint has a retention horizon - 2026-06-10 on this desk - which is why 44 of the 45 `activities report trades the executions endpoint did not return` days can never be repaired by retrying: the fills are gone from the API. The portal's activity export is not, and `scripts/journal_statement_import.py` reads it. **What the first real file measured**, and every design decision below follows from it: 974 rows, 884 of them trades, 133 trading days, 2026-01-02 to 2026-08-27, across both accounts; **zero unreadable rows**; and `Net Amount == Gross Amount + Commission` on **every one of the 884**, to the cent, with no exceptions. So the statement's single Commission column IS the complete cost, and splitting it into a guessed commission and a guessed fee would invent a breakdown the file does not contain - `fees` is written 0.0 and the total is exact. **The file is read with `zipfile` + `xml.etree`, not `openpyxl`**: an xlsx is a zip of XML, the sheet is one flat table, and a new third-party dependency is packaging trigger 1, owing a frozen rebuild for a format we can already read in fifty lines. Both the inline-string form Questrade emits and the shared-string form Excel produces on re-save are handled, plus CSV. **What a statement cannot say is the whole shape of the module.** (1) **No time of day** - every row is stamped "12:00:00 AM". Executions are therefore written at MIDNIGHT MARKET-LOCAL, and `journal_trade_shape.is_date_only` treats exactly 00:00:00 ET as "time unknown" (no fill can happen then; the market is shut and extended hours never reach it) so `session_bucket` returns None. Writing them at 09:30 to look complete would have tagged an entire imported year `opening_drive`; attaching the DESK's Pacific zone would land at 03:00 ET and defeat the check entirely. A date-only same-session round trip is a `day_trade` and **never a `scalp`** - zero elapsed minutes there is missing data, not a three-second trade. (2) **Fills are aggregated** - some descriptions say "AVG PRICE" in as many words. (3) **No execution id and no intraday sequence**: the statement's own row order is preserved and carried into the surrogate uid, because it is the broker's own listing, it is reproducible, and without it two identical fills on one day hash to ONE uid and half the position silently vanishes. A same-day round trip's LONG/SHORT label is therefore that ordering's claim rather than a measured fact - but the day's MONEY cannot be wrong either way, because a symbol that starts and ends a day flat realises the same total whichever way the legs pair. Per-trade attribution inside such a day is best-effort; the day total, which is what a tax return adds, is exact. (4) **Options carry a Questrade internal id in the Symbol column** (`8SVDLK9`) and the real contract in the Description; parsing the description into an OCC symbol is what keeps the expiry, the strike and the **100 multiplier** - trusting the Symbol column would make every contract its own opaque position and understate option P&L by two orders of magnitude. 174 of the 884 rows were options. **THE RULE THAT PREVENTS DOUBLE COUNTING: a statement never writes into a (broker, account, day) that a richer source already covers.** API rows carry real ids, real timestamps and unaggregated fills; statement rows carry none of those, and the two give the SAME fill different `execution_uid`s - so nothing in the upsert can see they are duplicates and importing both would silently double the position. `days_covered_by_richer_sources` refuses the day, at day granularity because that is the granularity a statement can be trusted at, and the count of refused days is reported rather than swallowed. `SOURCE_RANK["QT_STATEMENT"] = 1` is only the belt: a rank cannot compare rows it never sees together. **The one honest imprecision**: `rebuild_trades` recomputes gross P&L from price x quantity while Questrade books Gross Amount rounded to the cent, so the journal drifts from the statement. Measured on the real file: **-$0.1558 on $4,014.18 of realised P&L across 253 closed symbols**, worst single symbol 1.17c, worst single execution 0.5c, and **commission matched to the cent ($291.38 both ways)**. Immaterial, and stated rather than discovered later; making the assembler prefer the broker's own booked money is a change to the shared engine both brokers use and was deliberately NOT made in this packet. Coverage is marked COVERED only for days the import actually wrote trades into - a statement listing no trades on a day is not evidence none happened, since it may be a statement for another account. Account tax status is SEEDED from the Account Type column (`Individual TFSA` -> TAX_FREE, `Individual margin` -> TAXABLE) and never overwrites a label the trader set (I6), and an unrecognised wording stays unlabeled rather than guessed.
- **Auto-tagging has two lanes and they never compete** (trader request 2026-08-28: "i want auto tagging then I can come back and adjust"). The existing `journal_analytics.AutoTagger` matches a trade against the scanner's own output files - setup tracker, focus picks, AVWAP signals, intraday bounces - and answers "which of my setups was this?", which is the tag the trader actually wants. It cannot answer for imported history: those files hold the current lookback, not last February, so `suggest_for_trade` scores no candidates at all and a year pulled from a broker statement arrives as one undifferentiated untagged block. `scripts/journal_trade_shape.py` is the floor under that, deriving four facts from the trade's own row and legs - hold bucket (`scalp`/`day_trade`/`overnight`/`swing`/`position`, counted in SESSIONS via `market_calendar.is_session` so a Friday-to-Monday hold is one night and not three), entry session bucket, execution shape from leg ROLES, and instrument. It imports no scanner code, which is the boundary `AutoTagger`'s own docstring set and is worth more than the lines it saves; the five shared session-bucket names (`opening_drive`, `late_morning`, `midday`, `afternoon`, `closing_window`) and their cutoffs are therefore restated to match `bounce_bot_lib.learning.time_bucket_for` exactly, with `premarket` and `after_hours` added because a broker fills extended-hours orders and that module's `minutes < 60` branch would call an 08:00 fill an opening drive. **Three rules make the derived tags safe to average.** (1) **No tag is ever derived from the outcome** - no win/loss, no R, no "good_trade": a tag that encodes the result makes every per-tag statistic circular, and the `winners` bucket would post a 100% win rate that explains nothing. The outcome is the thing being explained and may never also be the explanation; the regression asserts a winner and a loser with identical shapes produce identical tags. (2) **Unmeasurable emits NO tag** - an open trade has no hold yet, an unparseable timestamp has no session, and a `SYNTHETIC_OPEN` leg means the opening fill was never imported so the entry shape is unknown rather than "one_and_done". (3) **Naive timestamps ATTACH market-local, never strip the zone off an aware one** - the same seam rule the adoption gate uses. Note what the store does upstream: `parse_broker_datetime` attaches the DESK's zone (Pacific) to a naive broker row, so a fixture written as "09:45" is stored `09:45-07:00` and buckets as 12:45 ET midday; the journal fixtures carry an explicit Eastern offset for exactly this reason. **Candidate ordering is by LANE, never by confidence**: shape tags are facts and carry 1.0, so a plain `ORDER BY confidence DESC` buries every setup match under `midday` - and the setup match is what the trader opened the pane for. The stored `auto_tag_summary` gives setup tags the first two of four slots and lets either lane spread into the gap, so a scanner-matched trade still reads as one and an imported one still says what kind of trade it was. **Accepting a suggestion drops that SUGGESTION, never the trade** - the 2026-08-24 reasoning that a tagged trade can still deserve a second tag is unchanged and kept; what changed is that a confirmed trade no longer re-proposes the tag it was confirmed with, which is how 220 proposals stood against one confirmed annotation and why the queue could not fall below the number of trades in it. Around it: a tag filter on the SHARED header (so one tag narrows the calendar, the equity curve and the fee totals, where Analytics could previously only group BY tag), `distinct_tags` counting the trader's lane separately from the machine's, and `rename_tag`, which rewrites `setup_tags` only - a derived tag is re-computed on every refresh, so the manager marks those rows and refuses them rather than accepting a rename the next rebuild would silently undo.
- **The Market Journal is what the trader thought; the Journal is what they traded** (R10.H, 2026-08-24). Two left-nav pages with near-identical labels, deliberately: `market_journal.jsonl` (`market_journal_entry_v1`) behind ONE service (`ui/services/market_journal_service.py`) used by both its surfaces - the Desk "Journal" tab after Capture (M5 default, Ctrl+Enter) and the left-nav "Market Journal" page. Merging it with the trade/tax journal would turn the tax record into a diary. **An entry is never backdated**: `session_date` is the session it is ABOUT, `created_at` is when it was written, and `written_after_the_session` is COMPUTED from the two so a caller cannot set it wrongly. Corrections supersede; the original stays on disk. Beside it, R10.G's `daily_market_context.jsonl` and the regime-shift stream are the MACHINE's half of the same day - every auto shift and every trader override, because the difference between them is the agreement rate the journal page shows. **Every entry now carries the tape it was written against** (trader rule 2026-08-27, after a day of notes rendered as an empty page): `scripts/market_journal_capture.py` stores the symbol's M5/D1 and SPY's M5/D1 as BARS - never pictures, which cannot be re-ranged, measured or read by the AI layer - in a per-capture sidecar, plus a short text digest (session range position, VWAP, prior-session extremes, 20/50/200 SMA, RVOL) in a `market_journal_chart_v1` row on stream `market_journal_charts`. `market_journal_entry_v1` is UNTOUCHED: a capture joins by `entry_id` from outside, which is what lets it be written AFTER the entry on a worker - a note must never wait on a chart, and a chartless entry is a smaller loss than a lost thought. Every bar list is a cache read (`AlertCenterPanel.journal_chart_bars`); nothing fetches. **An auto-mode flip writes its own row**: `AutopilotService.autoModeChanged` fires only when `auto_mode` actually moves (a profile change while Auto is OFF is not a flip), `MainWindow._record_auto_mode_flip` attaches SPY, and the row carries `ORIGIN_AUTO_MODE_FLIP` so `is_machine_entry` can mark it `[desk]` - one timeline, but a reader counting "what did you think?" never counts a sentence nobody thought. The two defects that produced the empty page are also fixed and must stay fixed: the page loads on `showEvent` (nothing called `reload()` at all), and BOTH surfaces use `shared_journal_service()` (the desk tab had built a second instance, so its `entryWritten` reached nobody). **`market_journal` is in `briefs.DEFAULT_SCOPES`** since the same day - the trader reversed R10.I's opt-in in as many words - while `TICKER_BRIEF_SCOPES` stopped being an alias and keeps the original four, because a session-level entry in a per-symbol packet is the TB-0/TB-5 failure mode.

  **WS-10D (2026-09-13, WISHLIST 10D/10K): the same store now tells the session's story and
  challenges the thesis in it — deterministically.** Three labelled kinds, never blurred
  (`scripts/market_story.py`): `trader_said` is the day's entries verbatim in `created_at`
  order, each carrying `written_after_the_session` and `predicts_this_session`, so the SAME
  sentence typed at 11:00 Pacific and at 21:00 Pacific is a prediction and a description and
  the pane says which; `measured` is arithmetic over COMPLETED daily bars for `SPY QQQ IWM
  VXX TLT USO` with `bars_through`, `bars_used` and a named rule version per number, and a
  benchmark with no bars is `unmeasured` with every field `None` and a reason naming it;
  `ai_said` is the third kind and is ALWAYS EMPTY here — code computes, the model explains in
  a later packet — and it exists as a field so no renderer has to guess whether a sentence
  came from a person or a model. A session with no note has an empty `trader_said` and a
  sentence saying so: no note, no invented thesis. **Phase 0.31 repaired the subject-session
  stamp.** `EvidenceLedger.append(subject_session_date=...)` preserves the session the note is
  ABOUT in `session_date` and records the market-local write day separately in
  `written_session_date`; the monthly segment follows the subject. `market_journal.session_of_entry`
  prefers that explicit pair and keeps the `created_at` reconstruction only for legacy rows,
  so a 21:00 Pacific review remains with the completed session and older evidence still reads.
  **The thesis quotes itself** (`scripts/market_thesis.py`, `market_theses.jsonl`,
  append-only, keyed on `entry_id` + `extractor_version`): every field carries a span that
  reproduces it exactly, `unstated` is a real answer with no span, and the invalidation and
  the condition are found first and BLANKED OUT of the text the stance is read from — an
  invalidation states the opposite of the claim by construction ("I expect SPY to hold above
  5,400 … if SPY loses 5,400 I am wrong"), so a stance counted over the whole sentence sees
  one bullish and one bearish word and gives up on a clear view. The horizon is counted in
  exchange SESSIONS: a "this week" thesis opened Tue 2026-09-08 is still OPEN on Mon
  2026-09-14 (four elapsed) and CLOSED on Tue 2026-09-15, where `timedelta(days=5)` would
  land on Sunday and close it a session early. A contradiction is a stance REVERSAL on the
  same benchmark inside the horizon, nothing subtler. A trader edit is a NEW row superseding
  the draft; the journal entry is never touched. An imported weekly forecast (10K) is
  `origin=external_forecast` plus a `kind=forecast` sidecar, shown under its own heading and
  never in `trader_said`, and an unsupplied creation time stays the literal `unknown` — a
  later import is not information known earlier. **A month is not the sum of its weeks**
  (`scripts/market_story_rollups.py`, the last deterministic nightly slot): the five ISO
  weeks touching September 2026 hold 24 sessions between them while September has 21, so each
  week belongs to the month of its THURSDAY, a month pack exists only for a month that owns a
  week, and the month adds its uncovered days afterwards (28-30 September live in October's
  week 40 and still come home). Week 37 of 2026 has FOUR sessions — Labor Day is the 7th —
  and is COMPLETE. Every pack names its covered, expected and missing sessions, carries the
  open theses forward, and is rebuilt only when its `inputs_hash` changed.
- **The scan cycle is timed, and the instrument must never become a scheduler** (trader-authorized 2026-08-25). Every call between the top of `run_strategy` and its "Monitoring N" line is silent on the normal path, so when the loop spent **12:55:00–14:27:36 inside one cycle** on 2026-08-25 — delaying the after-close sweep 52 minutes past its 13:35 due time — the logs could narrow it no further than "somewhere in the preamble". `ScanCycleClock` (module level in `bounce_bot_lib/legacy.py`, pure) now marks eleven stages and `run_strategy` logs **one** line per cycle, slowest first: `Scan cycle 41 preamble: 92.4s total: rrs_scan 88.1s, …, +8 other 2.3s`. Stages past the named few are **counted, never dropped**; a backwards clock reports 0.0s, never a negative stage. `_maybe_refresh_learning_after_close` logs when it first finds work due and once per worker while waiting, and stays silent when nothing is due. **It measures and formats and decides nothing** — a test parses the class and fails if it ever calls `sleep`, `wait`, `start` or `Thread`, because a timing helper that could defer or skip would be the scheduling change that was explicitly NOT authorized.
- **A sweep-finalized trade counts under the policy that MEASURED it** (trader Decision A, 2026-08-25 — `docs/archive/analysis/POST_ATTACK_AUTHORIZATION_2026-08-25.md`). The after-close sweep finalizes with a BLANK eod-hold `close_r` by design (`no_eod_close`): with no bars through the close there is no such number, and inventing one would make the same trade report differently depending only on what the finalizer held. What it did measure is in `context.exit`, not `context.path.exit_policies`. `setup_scoreboard.exit_policy_r` derives three frozen policies per final — `eod_hold` (the settled `close_r`), `stop_exit` (the STORED `context.exit.stop_exit_r`, read only where `exit.stop_hit`), `last_measured` (`(last_measured_close - entry_price) / risk_per_share`, sign flipped for a short) — as columns `r_eod_hold`/`r_stop_exit`/`r_last_measured`. **`usable` = at least one policy measured the row**, still ANDed with the risk floor and the R10.B claim split; unresolved rows stay unusable and are counted by reason. **They are never blended**: one table per policy through `evidence_stats`, and every eod-hold ranking view reads `r_eod_hold`, never `close_r`, so a row with no EOD close cannot widen an eod-hold n. Keying `usable` on `close_r` made all 656 of the 2026-08-25 finals invisible to every evidence surface; the live read moved 0 → 255 usable.
- **`unresolved` means UNMEASURED, and a swept trade that measured its bars is `swept_measured`** (M2, trader authorization 2026-09-05: *"Fix all of these failures"*). Decision A above got the ARITHMETIC right and left the LABEL wrong. `sweep_pending_bounce_outcomes` "needs no bars and no IB", so `finalize_outcome_once` is called with none and the writer's `status = "eod_complete" if basis == "measured" else "unresolved"` made every swept row read `unresolved` - the same word as a trade that measured nothing at all. Measured read-only over the live store, twenty sessions to 2026-09-05, 8,161 events over 324,605 rows: **measured_eod 3,820, measured_swept 3,607, unmeasured 644, open 90**. The 4,251 `unresolved` rows split 2,054 `last_measured_bar` + 1,553 `stop_hit_from_prior_measurement` (all 3,607 with `bars_elapsed > 0`, reason `no_eod_close`) against 644 with basis `unresolved` (284 `no_bars_after_entry`, 360 `no_measurement_in_checkpoint`). **63% of the mislabelled rows fall on 2026-08-24..08-27** - the F1 GIL-freeze days, when the live thread stopped scanning symbols through the close and the sweep finalized the backlog: 08-24 reads 0 eod / 466 swept, 08-25 2 / 636, 08-26 25 / 541, 08-27 38 / 663. **Those rows are NOT re-finalized.** They keep the R they measured and are read correctly by `outcome_semantics.terminal_kind`, which is the whole design: history is READ, never rewritten (plan.md sec 5). Whether a sweep running inside the same session could fetch the missing close bars from cache is a detector-side question and stays ask-first. The four kinds are `measured_eod` (bars through the close), `measured_swept` (the sweep settled it from bars measured earlier - it counts under `stop_exit` or `last_measured` and **never under `eod_hold`**, which it has no number for), `unmeasured` (nothing was ever measured; nothing about it may be averaged) and `open` (no final row). The writer's half is one function, `outcome_semantics.status_for_finalization_basis`, so the label and the reader cannot drift; the value is ADDITIVE (header unchanged, `schema_version` still 4, no reader in the tree enumerates the status domain). **The champion aggregator `_latest_bounce_outcome_rows` is the only production reader keyed on the status column and is deliberately UNCHANGED** - it takes `eod_complete` rows only, so `swept_measured` is excluded exactly as `unresolved` was and no tier, mute or PROVEN stamp moved. `setup_scoreboard` never loaded the column - which is exactly why **a row that claims to be `final` and carries neither a status nor a basis is decided by whether it recorded a close**: 13,703 of the live file's 14,863 `eod_complete` finals predate R10.A's `finalization` block and arrive status-less through the scoreboard's frames, so a numeric `close_r` or `eod_close` reads `measured_eod` (all 13,703 have one; **zero** do not) and the absence of both reads `unmeasured`. It is never `open` - the row said it was final - and a `status` cell reading `nan`, which is what a frame with no `status` column yields per row, follows the same rule rather than being taken for a status nobody registered. This reports what the finalization CLAIMED, not whether the number is usable: `unsettled_close_mask` still excludes the old `close_r == 0 and eod_close == entry` sentinel from every `eod_hold` mean. **749 pre-R10.A schema-1 finals** (`stop_seen` 397, `target2_seen` 166, `complete` 129, `stop_and_target2_seen` 57) do carry a status, an unregistered one, so they grade `unmeasured` - though `complete` at least was a measured outcome, which means an ALL-HISTORY report understates `measured` by up to 749 until those four are classified. They fall outside the 20-session window, so nothing live reads them today.
- **Evidence stores are never allowed to cost the thing they record** (R10.B-H). The Focus membership stream, the tracker transition ledger, the regime-shift rows and the journal all fail quiet: a failed append loses the event, never the pick, the tracker save, the regime change or the trade. The one exception is a journal WRITE, which reports failure loudly - a capture that did not reach disk must never look like one that did. Ground rule 10's statistics contract lives once in `scripts/evidence_stats.py` and every ground-rule-11 surface reads it from there; `outcome_semantics.claim_kind` decides what may be averaged as a trade at all, and **59% of the outcome store is annotations** that were previously averaged as trades.
- Chart paint lines (A4, landed on `testing` 2026-08-09): `scripts/chart_levels.py` builds the D1 S/R stores, prev-day H/L and the projected D1 trendline into a `levels` payload on the ChartDataService **worker** — never the paint path — and `CandleChart.set_levels` draws them with stable ids and click-to-select (`levelSelected`). One paint-lines control (`ui/widgets/paint_lines_button.py`) shows/hides groups, machine-local, defaults all-on. Trendline availability is surveyed in `docs/archive/D1_TRENDLINE_SURVEY.md`; measure it on the desk with `scripts/d1_trendline_survey.py`.
- Price alerts: the Focus tab and Research advanced view share one `PriceAlertService`; the desk polls and pushes fired alerts to the phone at ntfy `urgent`. The satellite relay/toast layer and the planned satellite edit intents went with Desk Link (removed 2026-08-24). The generic `read_only` mode on the price-alert board and panel SURVIVES the removal — it is a widget capability with its own tests, not satellite plumbing — and now has no production caller. See `docs/archive/FOCUS_PRICE_ALERTS_PROPOSAL.md` and `docs/EVENING_MODE_RUNBOOK.md`.
- Phone push policy (trader rule 2026-08-11, extended 2026-08-14): **AWAY is the only Auto mode that pushes routine output**, with **two** deliberate exceptions — the Research/Focus price alerts (every mode, including OFF) and EVENING's SPY ±1% wake alarm (`_maybe_push_spy_alarm`, urgent, repeating every 5 min while it holds, kill switch `push_evening_spy_alarm`). In AWAY the hourly swing push also carries the full favorite/high-conviction roster, and a second hourly push names the D1 level/event alerts since the previous one (the Alert Center classifies, `AutopilotService` aggregates and gates). Before adding any new ntfy sender, gate it on `auto_mode == AWAY` or state why it belongs with those two exceptions. The Price Alerts panel's **Test wake alert (urgent)** button (`PriceAlertService.test_push(urgent=True)`, added 2026-08-20) is a channel TEST, not a third sender: nothing schedules it, only that button calls it. It exists because ntfy has no Apple critical-alert entitlement, so urgent priority alone cannot override iOS Sleep Focus — the device-side steps are the Sleep breakthrough checklist in `docs/EVENING_MODE_RUNBOOK.md`.
- **The adoption gate compares timestamps at one seam** (`_gate_moment` in `autopilot_core`): the stored `gate_bar_end` is ALWAYS timezone-aware (it is the profile's `as_of`) while `gate_checked_at` and the caller's clock are naive `datetime.now()`. That mismatch crashed every adoption on 2026-08-19 and cost a whole session. Normalize by ATTACHING market-local to the naive side (`normalize_market_local_datetime`), never by stripping the aware side — stripping ends the crash and keeps the outage.
- **Movers-only chart review** (trader rule 2026-08-19, evening): a long inside yesterday's range is chop, so the chart-review queue shows only longs above the previous day's high and shorts below the previous day's low. It is a DEFAULT-ON PRESENTATION filter in `AlertCenterPanel._enqueue_review_alert`: it hides and counts ("N hidden (inside yesterday's range) - show", one click reveals for the session), never deletes, never mutes, never writes `review_policy.json` and never feeds the review-learning stream. UNKNOWN **shows**, tagged `unmeasured`. The deliberate Focus review (`review_focus_picks`) and armed chart-watch hits bypass it entirely. Focus surfaces flag a mover with the existing badge idiom (`MOVING`). The measurement is `focus_adoption_gate.mover_state` — the adoption gate's own extreme leg, no second copy of the rule. **Since 2026-08-27 (trader rule 2) it has BOTH legs and is asked at SHOW time:** `vwap_state` is the gate's `session_vwap_state` over `regime_pause_hold.session_levels` (cached M5, completed bars, never BounceBot's dynamic VWAP), `_review_chart_state` hides on EITHER verified leg (deliberately not the gate's UNKNOWN-before-CLOSED ordering - one measured reason to hide is enough), and `_advance_review_queue` re-measures the next candidate before showing it, because EPD reached the pane an hour after its flag under VWAP and fading. The revealed-for-the-session flag switches both checks off; a revealed name is badged `wrong side of VWAP`. **Rule 3 (same day) added a third leg for D1 recommendations only** (`is_d1` rows and `focus_d1_event` flags, never intraday): a long must be above its SMA200, a short below its SMA50 - `scripts/sma_trend_gate.py`, averages off COMPLETED daily closes (a preview / today-dated bar is excluded), price off the last completed M5 bar or else the last daily bar, badge `wrong side of SMA`. The scanner still emits trend-contrary D1 shorts; gating them at the source is a detector change and needs golden fixtures first.
- **Intraday alerts are a list beside the chart, not a queue in front of it** (trader rule 2026-08-27): `ui/widgets/m5_alert_bar.py` is the LEFT column of the desk, before the chart column (`TradingDeskPanel` splitter `bar | alert_center | setups`, three `desk_layout.DESK_SPLIT_*` weights; built in the middle and moved left the same morning on the trader's second pass, `DESK_SPLIT_KEY` bumped to v3), one line per alert newest first, `Copy all` (tickers, one per line, each once - a TC2000 paste) and `Clear all` (screen only), a click charts through `AlertCenterPanel.chart_alert` and removes the line (looked-at is done; the feed and History keep it). **Clicking from one row to the next is a SKIP, never a re-queue** (third pass, same day): `_select_review_alert` used to push every outgoing chart to the head of the waiting list, which refilled the D1 queue with the M5 rows the bar exists to keep out of it. `_current_review_holds_place` records whether the chart in front was POPPED off the queue (or is a clicked D1 row / armed hit) or merely clicked off the bar - a flag, not a re-test of the outgoing alert, because the same-symbol refresh branch swaps a queued D1 chart's alert object for that symbol's newer M5 one and `_is_m5_review_alert` would then drop a real queue member. Only a place-holder is re-inserted; the other writes a `skip` review event (dwell + `detail.reason = clicked_away_from_m5_alert`), because `_render_current_review` already emitted the `shown` impression and `shown` is the denominator for P(take | shown) - an unanswered impression would bias the rate. No parking: that stays specific to Skip-after-arming-a-D1. The routing is `_is_m5_review_alert` inside `_enqueue_review_alert` AFTER the AWAY branch - an ordinary intraday alert is emitted on `m5AlertPosted` and never queued; D1 rows, Focus D1 flags, chart-watch hits, armed price alerts, auto-pick proposals, typed symbols and the deliberate Focus review keep their chart. Everything upstream of that door (backing list, feed, History, evidence, AWAY recap) is untouched; the review queue is D1-only in practice. The queue-mechanics test files switch the routing off through one autouse fixture; `tests/test_qt_m5_alert_bar.py` owns it. **A click away IS a pass, and that is the intended meaning** (trader decision 2026-09-01, confirming the 2026-08-27 mechanic rather than changing it): *"clicking away = a pass. The tabs under the visual chart review should give us all the tools we need and we decide as we see. set alerts / add to focus and then move on."* So the `skip` row with `detail.reason = clicked_away_from_m5_alert` is not a shortfall to be repaired into a "take" or into silence by some later packet - it is the trader's answer to the impression, recorded. **The reason string is frozen**: `review_learning` keys on it, so renaming it silently re-partitions every cohort already accruing forward returns. Anything the trader wanted to keep from that chart they take with the tabs UNDER it (arm an alert, add to Focus) before moving on; those are separate writes and a pass never undoes them. Nothing in the code changed for this decision - only this entry and a one-line pointer at the writer.
- **The group RS/RW tape owns its own clock and reads 90 | 60 | 30 minutes** (rebuilt 2026-08-27 to `docs/archive/prompts/GROUP_TAPE_REBUILD_OPUS_PROMPT.md`, plan.md Phase 0.5 item 11; hidden earlier that day and shown again by the rebuild, live gate owed). It was never wrong, it was LATE: it refreshed only when a scan cycle's RRS pass finished (10-30 min apart, once 31 minutes late on a flip) and its one intraday number was a 60-minute `real_relative_strength` window off a 5-day fetch, so for the first hour it carried the overnight gap. Now: `scripts/group_rrs.py` is the formula, PURE and lifted out unchanged - a parity test feeds identical bars to it and to `legacy.real_relative_strength` and asserts 1e-9, and `SECTOR_ETFS` is a drift-tested COPY of legacy's map so the tape survives BounceBot being off. Its session filter is `completed_bars.completed_m5_bars` **plus a same-date filter**, which is the thing that stops a window reaching over the gap; `align_bars` intersects the two series on normalized stamps so a halted ETF cannot read as strength; 6/12/18 bars = 30/60/90 run off ONE filtered+aligned series and a window without `length + 2` bars is `None`. `ui/services/group_tape_service.py` is the Strength Board's shape - one QTimer, single-flight worker, last-good on failure, bounded shutdown - doing **ONE batched `yfinance` `period=1d interval=5m` download per 5-minute tick with no retry inside the tick** (Yahoo rate-limits bursts), quiet-hours gated with `refresh_now` exempt, **zero IB traffic and no `legacy.py` change**. A missing industry map means sectors only and no SPY bars today means "no read", both said out loud. On the strip an unmeasured window draws NOTHING (0.0 would claim "in line with SPY"), chips DIFF keyed by ETF and their variants live in `theme.qss` on a `side` property, and the callout carries the as-of and the status so a stale read is never silent. **The RS Window tab and `focus_picks_panel` still read `rrsSnapshotChanged` and must keep doing so** - it answers a different question (who led over the selected window at scan time). Not built, deliberately: industry as median member return (needs member bars - an IB-budget question).
- M5 Focus adoption gate (trader rule 2026-08-14, packet R2 — `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md`): one definition in `scripts/focus_adoption_gate.py` — an auto M5 Focus pick must be beyond yesterday's extreme **and** on the right side of session VWAP, measured on the last **completed** M5 bar, UNKNOWN always failing. Applied at candidate build, at every 30-min staging refresh (failures are evicted and may re-propose if they re-qualify), and at adoption via a stored verdict that expires after 45 min or 2 completed M5 bars, whichever binds first. Session VWAP comes from `chart_snapshot.session_vwap_series` — never BounceBot's dynamic/EOD VWAP, which blend prior sessions. On the AWAY/EVENING → DESK flip (packet R2.2) the drain adopts **only verdicts stamped after the flip**: the flip re-measures the queue, a failed re-measurement retries rather than falling through, and the 30-min refresh is the slower recovery.
- Focus provenance (packet R2): `focus_auto_picks.json` beside the focus files marks entries the machine adopted. **Absence of a marker means the trader owns it**, and only marked entries are reachable by "Not today" or the desync repair — that is how "user-entered names are never auto-removed" is structural rather than aspirational. BounceBot's triple-VWAP cut files a desync *request*; the Alert Center performs the removal, preserving one owner per store.
- Today's swing picks (trader-directed, 2026-08-31). The trader asked for it in as many words: *"at the end of the day I have a list of my top swing targets. I want a place to put them in so the bot knows my personal favourite picks. They will usually become focus picks too but these ones get special standing because I picked them by hand... put it at the very bottom of the M5 alerts tab, the tab is so long and I never use all of it. And the bot should scan the journal to know which ones I actually took."* Four things follow from that and are binding.

  **It is not the Master AVWAP like/dislike capture**, which already exists and records a verdict on a row the bot proposed. This records a name the trader brought in themselves, so it is stored on its own terms.

  **Two writes, in this order.** The swing Focus write-through goes first through the existing store (`FocusService.add(..., "swing")`), because that is the thing the trader asked for and it must not fail. The append-only evidence row (`swing_favorites.jsonl`, addressed by `project_paths.SWING_FAVORITES_FILE`) goes second, and a failed append is swallowed with a status line — an evidence store is never allowed to cost the thing it records. **Nothing in this chain calls `mark_auto_adopted`**: a hand-vetted pick with an auto marker on it would be reachable by "Not today" and the desync repair, which is exactly the removal path the marker exists to keep off the trader's own names.

  **A removal is a retraction, not an edit.** The add row stays where it is and a `remove` row follows it, so "added AMD and then thought better of it" and "never added AMD" stay different facts. The live list is a replay of one session's rows in file order; prior sessions stay in the store untouched.

  **The "taken" badge is display only.** It joins the day's picks against the TRADE journal (what the trader traded, `journal_feed`) — not the Market Journal (what they thought) — on symbol, marking a pick whose symbol has a trade opened on or after the pick date. It runs on a worker thread because the journal is sqlite over a year of fills, the window is bounded to 10 days because an unbounded query grows without limit, and it returns nothing when the journal would have to be created or migrated to answer: a display badge must never be the thing that triggers a schema migration. It derives no rate, grade or statistic — ground rule 10's statistics contract lives in `evidence_stats`, not in a chip.

  **SUPERSEDED 2026-09-14 — the placement only.** The trader moved the strip to the RIGHT column, under the Master AVWAP setups: see "D1C-L - M5 left, D1 right" at the end of this file. The two writes, the retraction, the `vetted` like-origin, the "taken" badge and the Copy/Paste seam below are unchanged; only the paragraph that follows is out of date.

  **Where it lives, and who decides how big it is.** The M5 alerts surface is a TAB in tabs mode and the tall left COLUMN in workspace mode, and the trader's saved setting is `workspace` — so the alert bar and the strip share one host (`TradingDeskPanel.m5_column`) that both modes mount, and the strip is the bottom of it either way. Nothing here touches `M5AlertBar` or any alert routing.

  That host is a **vertical `QSplitter`**, not a fixed stack — trader, same day: *"the tab needs to be resizable relative to the M5 alerts tab, I should be able to drag it up to see more."* It carries its own settings key (`qt_m5_column_split_sizes_v1`) so this drag and the desk's three-column drag never overwrite each other, and `setChildrenCollapsible(False)` because a strip dragged to nothing is one the trader cannot find again. The chip area therefore has a **floor and no ceiling**: a maximum height would make "drag it up to see more" do nothing past it.

  **Copy and Paste are the TC2000 seam.** Copy puts the day's tickers on the clipboard one per line, each once, in list order; Paste adds every ticker on the clipboard on the side the toggle is showing. Same idiom as the M5 alert bar's "Copy all", for the same reason.

  **What the rest of the system does with it.** Because each pick is a swing Focus entry, the human-focus tracker already grades it over 1/3/5/10 sessions — and the Focus like-origin is deliberately **`vetted`** rather than `manual`, so they form their own `human_focus_swing_vetted` sub-cohort and "how do my hand-picked swings do against the bot's?" is answerable from the existing grader. What is deliberately NOT built: `swing_favorites.jsonl` is not in `ai_summary`'s overnight evidence pack, and nothing joins it to per-setup journal statistics. Both are additive and unasked; `journal_analytics.AutoTagger` in particular reads the SCANNER's output files, and no tag may ever be derived from an outcome.
- Feed repetition control (packet R4 — `scripts/alert_repetition.py`): **display only, and it withholds nothing.** One live Alert Center row per symbol + side + market day; a repeat updates that row in place, keeps its first-seen time, gains an ×N badge and stays silent unless it escalates on a *strictly* higher tier, the *first* BANGER, or the *first* PROVEN (that list is exhaustive by trader decision). Ordinary alerts in the first 30 min after the open (`alert_open_digest_minutes`, 0 disables) join one digest row. **Focus-privileged, trader-armed, entry-assist and ready-D1 output bypass both the fold and the digest** — checked first, before anything else. The backing alert list is written *before* any repetition decision and never consulted by one, so History, the evidence streams and the AWAY push are untouched; every failure path falls open to a plain new row. No suppression field exists here or anywhere in this chain.
- One completed-bar rule (packet R5 — `scripts/completed_bars.py`): `bar_start + bar_minutes <= now`, **inclusive** at the boundary (a strict `<` discards the bar that just closed), timezone-converted with `astimezone` and **never** `replace(tzinfo=None)` — that spelling discards a stamp's offset instead of converting through it. `weekend_strength` delegates to it. BounceBot's ad-hoc copies at `bounce_bot_lib/legacy.py:4384-4386, 4533-4535` still use the wrong spelling and migrate *opportunistically*, never as a silent change to a shipped detector.
- Pure indicator modules (packet R5 — `scripts/indicators/`): `smi.py` (TC2000 parity — numerator and denominator smoothed **separately**, divided last), `efficiency_lrsi.py` (TC2000's "LRSI", 0–100; **not** the unrelated Ehlers `laguerre_rsi.py`), `heikin_ashi.py`. Completed bars in, immutable tuples out, `None` for anything unmeasurable. **No importer yet — the first one fires the packaging trigger.**
- M5 Strength Board (packet R2): `scripts/strength_scan.py` (pure formula) + `ui/services/strength_board_service.py` (single-flight owner, 15-min refresh on the quiet-hours window, last-good on failure) + `ui/panels/strength_board_panel.py`. Batched yfinance `period=5d` over `universe_all.txt` — **zero IB traffic**, so the locked pacing budget is untouched. Every board add re-runs the adoption gate at click time. Since 2026-08-19 every column is click-to-sort (presentation only — the sort never calls the service, so it cannot refetch; blanks sort last in both directions; Qt `setSortingEnabled` is deliberately off because the row buttons are cell widgets) and selecting a row charts it — **since 2026-08-31 in the desk's Visual Alert Review pane** (trader: *"when I click on a stock in this M5 strength board it should come up on the Visual chart review in the trading desk"*), previously the snapshot popup, which was right while the board was a page elsewhere and is a window in the way now that board and pane share a column. The click goes through `chart_symbol`, the **lookup box's** door, and deliberately not through `_enqueue_review_alert`, the SCANNER's door, which would have been wrong four ways for a click: it drops everything in AWAY, drops parked symbols, diverts M5 alerts to the alert bar, and can hide a row behind movers-only. A name the trader clicked must appear. It charts as a `MANUAL_CHART` — muted, not red; nothing fired, the trader was looking — never enters the alert feed, and carries its side so a short is not charted as a plain WATCH. Still no second chart widget anywhere. **Since 2026-08-31 it is not a page at all.** The trader: *"The Strength Board tab is good but it really should be modified to fit in the 'strength' window in the trading desk — either integrated directly or be positioned below it."* It is now a collapsible section under `FocusStrengthBoard` in the alert column, hosted by `AlertCenterPanel.attach_strength_board`; `MainWindow` still builds, owns and (new) shuts down the one service, so only the wiring moved. The section **starts closed**, because the alert column has a 360 px floor and everything left of it is chart: closed it costs one header row. Three width facts drove the build and are the reason it does not squeeze the charts — a `QToolButton` demands its whole label (315 px measured for this title under `theme.qss`), so `CollapsibleSection`'s header is Ignored horizontally and elides; the board asks 270 px, so it is hosted in a `QScrollArea` and that minimum stops there instead of reaching the desk splitter; and the status label wraps because it carries failure reasons and unwrapped asked for 434 px. The two sides stack **vertically** now — side by side was right for a full-width page and is unreadable in a column.

- **The Strength Board's relative volume is SESSION-RELATIVE, and the offset is counted inside its session** (V1 2026-09-02, corrected by R4 A7 on 2026-09-02). Decision 0016 answer 9 spells the trader's formula as `AVG(V / mean(V at the same bar offset over the prior 15 sessions), 12)` and the same answer calls it "the time-of-day relative volume". V1 implemented a flat POSITIONAL stride - `V78`, `V156`, ... `V1170` - and defended it in the module's own docstring on the grounds that TC2000 is positional and parity with the trader's scan is the requirement. **That reading does not survive one short session.** A half day is 3.25 hours, 39 bars; a single one of them anywhere inside the sixteen-session window shifts every offset past it by 39 bars, so a 10:00 bar is compared with a 13:00 bar and the number silently stops being the thing its own name says it is. Measured on a synthetic series whose volume is a pure function of the time of day - the one case where the answer must be exactly 1.0000 - one early close made the positional stride read **1.2949**. So `strength_scan.relative_volume` groups the bars by session, takes each of the last twelve bars' offset WITHIN its own session, and averages the volume at that offset over the fifteen prior sessions. **A session that never reached that offset contributes NOTHING rather than a zero**: an early close is missing evidence, and a zero in the denominator's mean would read as "that day was dead at 10:00", which is a claim about volume rather than about the calendar. "Not enough history" is counted in SESSIONS for the same reason - fifteen prior sessions or the cell is blank. The residual is stated rather than hidden: the offset is the bar's INDEX, so a session missing a bar in its MIDDLE has its later offsets shifted by one; measured at seven basis points on the golden's `GGG`, against the 29% a 39-bar early close cost the positional stride. Keying on minutes from the open would remove even that, and it is a different rule from the one the trader stated, so it is not made silently. **`relative_volume` is deliberately NOT one of the seven fenced formula functions** (`sma`, `displaced_close`, `true_ranges`, `atr`, `strength_score`, `percentile_cut`, `ema`), which stayed byte-identical to the R8 baseline through this change - that is what the narrowed fence is for. The golden `tc2000_parity_v1` gained two symbols with this: AAA-EEE are clean 78-bar sessions on which both readings agree, which is exactly why the original five could never have caught it, and `FFF` (one early-close session) and `GGG` (one missing bar) are the cases that can. Its expected values are still computed by a second naive implementation in the builder, written from the trader's line rather than from the module under test. **The D1 floors read CLOSED bars over a WIDER window** (R4 A8): the daily download had no completed-bar filter at all, so today's FORMING daily bar went straight into the 100 and 200 SMA and the floor a row was greyed against moved on every refresh - most at 09:31, when today's "close" is nine minutes of trading. `market_calendar.last_completed_session` decides; a calendar refusal falls back to dropping a row dated today rather than blanking every floor on the board, because that would be a much larger claim than the one being avoided. `DAILY_FETCH_PERIOD` is `2y`: `1y` is about 252 sessions against a 200-close requirement, and those ~52 sessions of slack are what a listing date, a provider gap or a holiday run spends - on exactly the names most likely to be interesting. And `autopilot_core._frame_rows` coerced a missing volume to `0.0`, which reached this relative volume as "this bar traded nothing"; it passes `None` through now, so the cell is blank. A genuine zero-volume bar is still data; a negative one is not a quantity and joins the blanks.

  The **RS/RW half** the page carried from 2026-08-21 retired with the page. It existed so the two reads could be compared without flipping PAGES; the Alert Center's own RS/RW Board tab is now one tab-click away in the SAME column, so keeping it would have been two views of one payload six inches apart. The tape, its owner, the `rrsSnapshotChanged` signal and that tab are untouched — one listener retired, nothing else moved. If the trader wants that second view back it is a section, not a page.
- Auto-mode matrix (trader rules 2026-08-14, packet R1 — `docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`): discovery is identical in every mode; what changes is who is present. **DESK** adopts staged auto picks into M5 Focus immediately. **AWAY** stages and never adopts. **Since the R1 trader amendment 2026-08-24 (BUILT, canary owed) it does NOT accumulate a chart-review queue** - a full AWAY day once left 317 pending review items, and the return surface is now the EOD recap (left-nav "AWAY Recap") instead. Discovery is unchanged, and the backing alert list, History, the D1 badge and every evidence stream still fill exactly as before: the routing sits at `_enqueue_review_alert`, the single door into the queue, and everything upstream is written before it. The AWAY hourly phone pushes are unchanged (resolved sub-decision). EVENING keeps its queue. Sound suppression while AWAY is unchanged; the old drain-into-the-review-queue on the AWAY→DESK flip is exactly what the recap replaced. **EVENING** runs the open+30 early slot, the 07:00/07:15/07:30 strength checks and the briefing, then stops — no ordinary hourly slot, no open watchlist self-build — stages picks for the wake-up flip, and queues alerts silently on the same rule as AWAY (the trader is asleep; the SPY alarm is the deliberate wake channel). **OFF** does nothing automatic at all — no slots, no watchlist build, no sweep, and **no auto-pick adoption** (R1 spec §1 matrix; this line previously claimed the opposite and was corrected 2026-08-19). EVENING leaves the BounceBot sweep **running** — settled by the trader 2026-08-15 (that spec's §9): "no new scans" meant the scheduled swing scans and watchlist builds, not the sweep, which fills the alert queue, feeds the strength checks and already pauses itself at close+30.
- Quiet hours (packet R1): every **automatic** starter is gated on `autopilot_core.auto_scanning_due` — weekdays, session open−30m through close+60m (06:00–14:00 PT), fail-open on a session lookup it cannot answer. It covers the launch/tick universe self-heal, the boot resume that used to connect BounceBot to IB at any hour, the daily 07:00 self-arm, the open watchlist build and the swing slots. The window is deliberately a **superset** of `bouncebot_scan_window`; keep it that way or the gates contradict each other. **Manual buttons are never gated** — `force=True` is the carve-out on the universe rebuild.
- Auto/Away phone output: `autopilot_today.txt` is the single verified home-folder digest, with the safety/freshness header first, then numbered best swing trades, then intraday and condensed operations. Mode changes (OFF/DESK/AWAY/EVENING) are made on the main desk.
- Unattended: the separate mini-PC scanner role is retired (2026-08-08) — the 8845HS main desk is the only always-on machine and the only scan host, so no cross-machine IB budget question exists. `scripts/master_avwap_mini_pc.py` was **removed 2026-08-24** (P1.5); the named-slot scheduling shape it established lives on in `ai_jobs/runner.py`, which says so.

---

## Focus picks alert on PULLBACKS only (2026-09-01, Phase 0.12 A1)

Trader, 2026-09-01: the Focus D1 feed had become unreadable, and what filled it
was the extension half of the automatic event set - the "still going" news about
names the trader had already seen.

The 2026-08-05 rule (FRPT printing a new 20-day high and then simply staying
extended: *"it comes up as a new 20 day high alert but now it's extended and I'd
only want to see it on an SMA bounce or something"*) rationed those to one per
name per day. A ration was not enough. Every Focus name is now implicitly
watched for the PULLBACK set alone - a 15EMA reject, an AVWAPE or 1σ bounce -
and for nothing else.

**The gate is at the flag-GENERATION seam, not a filter downstream.**
`_poll_focus_d1_interest` builds `pending_kinds` from `D1_PULLBACK_KINDS`, so an
extension kind is never constructed, never evaluated, never flagged and never
has to be suppressed. That matters because the downstream chains in this desk
are display-only by rule and have no suppression field; the only honest place to
stop an alert is before it exists.

**Arming is the one surviving route, and it is a DIFFERENT poll.**
`_poll_d1_event_watches` reads `d1_event_watches.json` and is untouched by this
rule. Keeping the two lanes disjoint is what makes double-firing structurally
impossible: the automatic lane cannot emit an extension kind, so an armed one
can only arrive once.

The one-extension-per-day bookkeeping (`_focus_extension_spent`) is gone. It had
nothing left to ration, and a filter that can never fire is worse than no
filter - the next reader would take it for a live rule.

**Nothing else moved.** The prev-day break gate, the once-per-kind-per-session
registry, the window that opens AT the break rather than at midnight, the
`focusBreakStatesChanged` emission and the feed/beep routing are all unchanged.

## An armed alert has a life, measured in sessions (2026-09-01, Phase 0.12 A2)

The Armed inventory is supposed to read as "the exact conditions I am waiting
on". It accumulated forever, so half of it was a watch armed weeks ago on a
thesis that had since gone stale, and the surface stopped meaning anything.

**The windows.** A manually armed 5-day extreme watch gets 5 trading days; a
20-day one gets 10; everything else armed - D1 level watches, any-bounce
watches, manual price alerts - gets 10.

**Sessions, never weekdays.** `market_calendar.trading_days_between` is the
clock. Weekday arithmetic gets this wrong twice: it counts Thanksgiving as a
day, and a five-session watch armed on a Friday would come due the following
Friday rather than the Friday after.

**Uncertainty never deletes.** `armed_alert_expiry.is_expired` returns `None`
when the calendar refuses - a date outside its validated range, an unreadable
stamp - and `None` is never read as `True`. Every caller here removes something
the trader created by hand, so failing closed is the only safe direction.

**Nothing is silently lost.** Every expiry appends one row to the
`armed_alert_expiry` evidence stream naming the store, the symbol, the kind,
when it was armed, when it came due and how many sessions it was given. The
append is best-effort and swallowed on failure - an evidence store is never
allowed to cost the thing it records.

**A price alert is DISARMED, never deleted.** It leaves the Armed board, which
is what the trader asked for, and keeps its levels, its note and its trigger
history exactly where they were, so re-arming is one click and nothing has to be
retyped. That also keeps plan.md sec 5's "user-entered names are never
automatically removed" literally true of `price_alerts.json`, which the module's
own docstring had promised since it was written. **Arming restarts the clock**
(`price_alerts.mark_armed_now`, called from every arm site in the board) or the
re-armed alert would be disarmed again by the stamp that expired it.

**No new timer.** Expiry runs at the head of the poll that already owns each
store: the 60 s D1 watch tick for the three chart-watch stores, and the price
alert service's own poll for `price_alerts.json`.

**An entry with no stamp gets TODAY.** Never an older guess - guessing backwards
would disarm the trader's whole board on the first load after the upgrade.

## A Focus pick that never speaks fades (2026-09-01, Phase 0.12 A3)

A Focus list only means "the names I am watching" while something takes names
off it. A pick that has fired no alert and printed no pullback event for ten
trading days is not being watched; it is furniture, and it is what makes the
list too long to read.

**The clock.** It starts at ADD time and lives in `focus_pick_clocks.json`, a
sidecar `FocusPickStore` owns beside the focus files. Activity RESETS it: a
fired Focus D1 flag, an armed-watch hit (every armed poll builds its alert
through `_chart_watch_alert`, so one call covers all three lanes), or the
trader's own "★ keep" on the review chart - the strongest statement of interest
the desk ever gets. Ten trading days without a reset and the pick fades.

**It applies to the trader's own names, and only here.** Fading a hand-typed
pick is an explicit trader authorization given on 2026-09-01. It is scoped to
Focus and goes through the store's own removal path, so `_uninject_from_shared`
still refuses to touch a broad-watchlist line Focus did not inject
(CandidateRegistry invariant, plan.md sec 5). No other automatic path gains the
right: `remove_if_auto_adopted` still refuses anything without a marker.

**Faded is not deleted.** The pick moves to `focus_faded.json` with an
append-only row in `focus_fade_events.jsonl` behind it, and the trader gets it
back with "★ Restore to Focus" (a FRESH ten sessions - a restore is not a
fade-proof) or clears it with "✕ Discard", which leaves the evidence and only
clears the list.

**A faded swing favorite gets a RETRACTION, never an edit.** `swing_favorites`
is append-only by design - "added on the 3rd, faded on the 17th" stays two rows
in the order they happened - so the fade appends `ACTION_REMOVE` with origin
`focus_fade` rather than `trader`, because the trader did not do it and a store
whose rows all claimed to be theirs could not answer "did I drop this, or did it
time out?". The Focus entry is already gone by then; this writes evidence only.

**No pick-feedback verdict is written.** A fade is the desk noticing silence,
not the trader passing a verdict, and every verdict in `pick_feedback.jsonl`
feeds a graded surface. Inventing a "faded" verdict would put the desk's own
housekeeping into the trader's scoreboard. The membership `left` event carries
reason `focus_fade`, which is where that belongs.

**Uncertainty never fades**, on the same rule as A2. A clock the calendar cannot
read keeps the pick; a pick with no readable clock is re-stamped TODAY rather
than faded on a guess.

**Where it runs.** The day roll (the fade clock is measured in sessions, so that
is exactly when a pick can come due) and a half-hourly timer. Never inside the
60 s poll's per-symbol loop: it walks every Focus entry and asks a calendar.

**The faded walkthrough uses the ONE door.** `review_faded_picks` enqueues
through `_enqueue_review_alert` with `FOCUS_FADED_TAG`, which bypasses
movers-only exactly as `FOCUS_REVIEW_TAG` does - a faded pick is by definition
one that has not been moving, so the filter would hide every row of the list the
trader just asked to see.
## Grading what the trader already said (packet P1, 2026-09-01)

Four rules, all on the evidence side. None of them reaches a detector, score, alert,
watchlist, Focus list, review queue or `review_policy.json`, and none of them may be
allowed to cost the event it records.

- **A human-focus pick is identified by its category as well as its name.**
  `human_focus_tracking._pick_key` is (trade_date, symbol, side, category slot), and the
  slot is the base source with any like-origin suffix removed - so `focus_swing_vetted`
  and `focus_swing` are ONE swing membership and a re-snapshot under a newly-recorded
  origin adds no row. Without the category, a name already on one list swallowed its row
  on the other: on 2026-09-01 AMGN LONG was liked into swing Focus with origin `vetted`
  at 11:33:06, the day already held a `focus_m5` AMGN LONG row from 08:02:14, and
  `human_focus_swing_vetted` had **zero rows in all 4,083**. `focus_membership_events`
  had already diagnosed this (audit F3) and keyed its own episodes by category. **Any
  join over these files must use `pick_source_family`** - `weekend_prep_panel` does, or
  it would hand one category the other's forward returns. **A walkaway replays ONE
  position per (date, symbol, side)**: which list proposed a name is a cohort question,
  not a second position.
- **A like and a veto merge into their cohorts on the same click, through one helper.**
  `commit_like` and `commit_veto` both call `_merge_cohort_safely`, so the two cannot
  drift - they are read side by side on Weekend Prep and a difference between them has
  to come from the data. The like was nightly-only until 2026-09-01:
  `like_cohort_picks.csv` was last written 2026-08-27 against likes recorded through
  09-01. Failure is swallowed to a "(cohort update deferred)" status suffix because the
  annotation row is already on disk when the merge runs, and both merges are idempotent,
  which is what makes running at capture time safe.
- **A pre-versioning veto pools with the version that INTRODUCED its code**, never with
  the lowest version overall. `compressed` arrived in v2, so gating the unversioned
  mapping on `min(versions)` stranded its three pre-versioning picks:
  `human_focus_veto_compressed` (n=3, PF 165) read beside
  `human_focus_veto_v2_compressed` (n=18, PF 0.39) - one judgement as two opposite ones.
  Pooling stays a reading of the record: it happens only in
  `_rebuild_pooled_performance` and no pick or outcome row is ever rewritten. **Never
  assert a literal `vocab_version` in a test here** - load the vocabulary and discover
  the late codes.
- **The scoreboard grades every explicit decision, and the `r_gap` callout is
  report-only.** An action enters `TAKE_ACTIONS`/`REJECT_ACTIONS` on what its WRITER
  does, not on its name: approve writes a watchlist, remove calls `remove_everywhere`,
  `veto_day_trade` vetoes the D1 chart that was shown (its M5 interest is a different
  claim on a different timeframe). Machine events, `*_fired`, `*_expired` and every
  `disarm_*` stay out - none is a verdict on a chart. `r_gap` fires on the R difference
  alone, never the take rate, which is the only way to see a segment taken at the normal
  rate whose two halves measure far apart; it lives on the state and in the report and
  is deliberately absent from `draft_policy_from_state`, `review_guidance` and the AI
  evidence package. **Coded vetoes annotate the `dislike_reason` dimension and never
  re-resolve an episode** - the verdict comes from the review event store alone, a veto
  whose side disagrees is skipped rather than guessed, and a veto with no episode is
  left alone rather than inventing an impression.

## Phase 0.13 - the four rules CLAUDE.md gained and this file did not (added R2)

CLAUDE.md's `Core loop / data flow` promises that the incident, measurement and
trader conversation behind every rule is preserved here verbatim. Four rules
landed in Phase 0.13 without their entry. These are those entries.

- **The LRSI M5 alerts are retired and every row of their evidence is kept**
  (`bounce_bot_lib/legacy.LRSI_M5_ALERTS_RETIRED`, P0, trader 2026-09-01:
  *"LRSI alerts seem to be mostly spam. however I enjoy them as something that can
  boost the potential of an alert. for now let's put them on the back burner.
  let's measure how they perform on different timeframes but no need for their M5
  alerts."*). ONLY the GUI leg goes. The obvious implementation - flipping the
  entry in `M5_SIGNAL_TYPE_DEFAULTS` - was verified and REJECTED: that toggle is
  tested before the event joins `hits`, so it would have stopped DETECTION and
  taken the outcome rows with it. The retirement sits at the emit seam, after
  `record_alert_tier`, so the candidate row, `intraday_bounce_outcomes.csv`, the
  learning tier and the PROVEN stamp all keep running - which is what the
  "different timeframes" measurement the trader asked for is built on. Unlike the
  H1 retirement beside it, `log_bounce_to_file` still runs. Un-retiring is one
  constant, and the lane's tests monkeypatch it to False so the consequences are
  already pinned (R1).

- **A third auto-tag lane offers what the trader already SAID, and it is not a
  link** (`journal_analytics`, P6). It matches the trade's OWN window - open date
  to close date, never the fuzzy 16-day neighbourhood the scanner lanes search -
  and outranks every fuzzy source; a rejection is PREFIXED (`vetoed:`, `passed:`)
  so it can never read as an endorsement, and a pass carries ALL its codes in
  vocabulary order (R2 - `codes[0]` had been throwing the rest away, which made a
  two-reason pass into a different statement). `context_row_id` is a POINTER FOR A
  READER: plan.md P5.3/P5.4 own the canonical opportunity id and a second one must
  never be invented; only 54 of 730 take-class review rows carry an alert
  `event_id`, so the rest point at their own natural identity.
  **A chart housekeeping action is a LINK, not a tag.** `add_focus`, `arm_level`,
  `arm_watch` and the toggles say the trader did something WITH the chart and name
  no setup; 676 of 730 live rows carry no `bounce_types`, so the lane minted
  `took:<action>` for almost all of them and - ranked first, at 0.90-0.95 - spent
  a slot of the four-slot Tags column on it. Measured: EYPT and SMPL lost
  `avwape_to_1stdev` to a housekeeping click, and on the bulk tagger TRV lost
  `avwap_retest_followthrough` at 0.91 to `link:review:arm_level` at 0.95. ONE
  predicate (`is_link_candidate`, accepting both the in-memory flag and the
  `link:` prefix that survives the store) now rejects them in the summary, the
  bulk lane, the bulk top pick, Accept/Accept-all and `tag_confidence`. They still
  RENDER, with their event id: the pointer is worth seeing, it is just not a tag.

- **The trader owns `trade_annotations`, and there is exactly ONE machine writer**
  (`journal_bulk_tag`, P6a, trader 2026-09-01: *"let's get Opus to do the tagging
  and I can review after"*). 193 trades and ONE trader-typed setup tag is what
  prompted it. `tag_status` is `confirmed` / `provisional` / `needs_review`, and
  the column's DEFAULT is what made it safe on a live database: every existing row
  was typed or accepted by the trader, so it became `confirmed` the moment the
  column appeared and no backfill had to decide that afterwards. The refusal to
  overwrite a confirmed row lives in `JournalStore.apply_provisional_tags`, NOT in
  the caller - an exception that depends on every caller remembering a rule is not
  a boundary. It never promotes a shape tag (a fact about the clock at confidence
  1.0 would outrank every scanner match while answering a different question) and
  **never writes `tag_corrections`**, because that table is the trader's feedback
  TO the tagger: only an EDIT teaches it, and agreeing with a guess would raise
  that guess's own confidence forever. Below the threshold it writes NO tag, only
  a marker - a low-confidence guess in `setup_tags` would be counted by every
  per-setup statistic, which is the circularity the tagging rules forbid. The
  threshold, 0.70, encodes a sentence rather than a percentile: "the tracker or a
  focus favourite named this symbol, on the day I traded it, on the side I traded".
  Run 2026-09-01: 24 applied, 132 marked, 0 refused, 0 corrections written.
  "My setups" counts CONFIRMED tags only, over DISTINCT closed trades (R1 - summing
  the buckets of a non-exclusive group measured 24 of 156 as 40% and suppressed
  the very note that exists to say how thin it is).

- **The setup registry is frozen DATA and is not authoritative yet**
  (`setup_registry`, P7). Five naming sites, one entry each, keyed
  `setup_id@version`; `legacy.py`'s `*_STUDY_FAMILY` constants are the fifth and
  eight families are named ONLY there, so a registry built from the four sources
  the packet listed would have omitted detectors that run every scan. It is
  regenerated deliberately and reviewed as a DIFF, never rebuilt at import: a
  crosswalk that recomputes itself from five moving sources is a sixth source, and
  its disagreements would appear and vanish unseen. It RESOLVES NOTHING - eight
  `known_divergences` record what each source believes, because choosing which
  spelling is identity is a decision (P4.1's) and not a derivation - and FILLS
  NOTHING its sources do not establish, because a guessed `supported_sides` reads
  as established in exactly the column a later experiment trusts. An unresolvable
  name RAISES: a silent fall back to `GENERAL` would file "two tables write
  different things under one word" under "untagged". Its sibling
  `research_warehouse/trial_ledger` writes one row per registered grid BEFORE any
  outcome is inspected, refuses to rewrite a `trial_id`, and stamps `registered_at`
  - an undated declaration and one written afterwards are indistinguishable six
  months later, which is the whole thing the ledger exists to rule out.

## P9 - the quick like, and the sidecar that finishes after the close

- **One key says "something about this was good", and that is the whole verb**
  (`capture_rail.commit_quick_like`, Alt+L, trader 2026-09-02: *"anytime I like
  and claim a setup or like a day trade setup I just want to let the bot and the
  future AI know 'something about this was good' and then we can figure out what
  about it / what's the best entry later."*). It writes `like_claim` with
  `like_mode: "quick"`, no claim and no why. **This supersedes R9.2(a)'s "a like
  needs a why" for the QUICK path only** - the claimed path is untouched, and its
  why is still required for the reason it always was: 31 dislike strings were
  lost to a field nothing insisted on, and a claim nobody can check later is the
  same mistake with a label on it.

  Alt+L was chosen because it is UNBOUND: the whole inventory in `scripts/ui` is
  Ctrl+F, Ctrl+J, Ctrl+R, Ctrl+Return, F9, Alt+E and the rail's Alt+V/K/N/P. Two
  live bindings for one sequence is an ambiguous shortcut and Qt fires NEITHER,
  so a clash costs the trader both verbs silently.

  Everything a claimed like does to the review, this does - and none of it needed
  code: the chart RETIRES and `like_advance` is recorded because both are keyed
  on the event type, and the symbol is marked reviewed today because `like_claim`
  was already in `_ANNOTATION_DECISIONS`. Everything a like has never done, it
  still does not: no Focus, no park, no watch, no alert, no watchlist. A LIKE
  CARRIES ZERO PRIVILEGES (plan.md P3.1), and a one-key verb is worthless if the
  trader has to wonder what else it did.

  `like_mode` is ADDITIVE and the schema version stays 1. That is proven, not
  asserted: a test hands the loader, the like cohort, the auto-tagger's capture
  lane and the pass cohort a row carrying the new key and each returns its normal
  answer. A row written before P9 has no `like_mode`, and absence reads as
  `claimed` - a claim was REQUIRED until this packet, so there is no other
  possibility. `store.like_mode_of` is the single place that says so.

  A quick like grades under `like_unclaimed`, where an unnamed like already went.
  It contributes a LINK to the auto-tagger - a pointer with an event id and NO
  tag text - because it names no setup, and "liked" in a Tags column would mean
  nothing about the setup while outranking the scanner match beneath it (R2).

  **The key and the button are two verbs, on purpose** (trader, 2026-09-02:
  *"ensure we also just have a button on the visual chart as well. Maybe it can
  have a pop up with a note I can put in similar to what we have in master
  avwapsetups"*). **Alt+L stays instant** - a key that stops to ask a question is
  not a one-key verb, and the whole value of the shortcut is that it costs
  nothing. **The button opens a box** for an OPTIONAL note, using
  `QInputDialog.getMultiLineText`, the same control the setup tracker's dislike
  detail uses, so the gesture is already familiar. OK with an empty box is a
  plain quick like; CANCEL records NOTHING, because a dialog that wrote a row on
  cancel would be unusable for "let me look at this first".

  An optional note is NOT R9.2(a)'s required why returning: that rule requires a
  reason for a CLAIM, and this path makes none. There are two buttons and one
  implementation - the chart's calls the rail's `prompt_quick_like`, because the
  capture rail owns capture and a second route to the same write is a second
  thing to keep in step. On the chart it is APPENDED to the existing verb row:
  still ONE row between the charts and the tab strip, and every button that was
  already there keeps its spot.

- **A capture sidecar is finished after the close, and the original is never
  rewritten** (`ui/annotations/sidecar_completion.py`, nightly slot
  `sidecar_completion`). `pass_cohort`'s intraday columns were blank on EVERY
  live pass, with the reason `sidecar_ends_before_the_entry_bar`. That was not a
  defect in the grade: the sidecar holds the bars the desk was ALREADY HOLDING at
  the click, so the entry bar the rule asks for - the first completed M5 close
  AFTER the click - is by construction never inside it. Gate 34 recorded this as
  an open definition question (should entry be the last close AT the click?).

  It does not have to be. The rest of the session exists after the close; it was
  simply not in the desk's hands at the moment the key was pressed. The slot
  appends those bars from the research lake - narrowed ARROW-SIDE by symbol and
  interval range through `read_rows`, never a materialised list (BD-74) - or from
  the desk's own bar cache when the lake has not ingested that session yet, which
  is the normal case the morning after.

  **The completed bars go to a NEW file and a NEW field**
  (`<event_id>.completed.json`, `m5_bars_completed_ref`). The row's original
  `m5_bars_ref` keeps meaning "what the desk was holding at the click" - a fact
  about that moment, not ours to edit - and the two together show exactly how
  much of the session the trader could actually see. One reader
  (`read_completed_bars`) prefers the completed file and falls back to the
  snapshot, so no grader has to remember which to open; remembering is what
  produces two graders that disagree.

  Idempotent, fail-open, and every refusal counted by its own reason: no research
  store, an unreachable share, no bars anywhere, already complete, already
  completed. An unfinished sidecar is a gap; one padded from nowhere would be
  worse than a blank grade. The slot sits BEFORE `pass_cohort_grading` because it
  feeds it - the same night completes and grades, rather than the morning after.

- **That read is TIMEZONE-AWARE, and a failed read names itself** (packet N1,
  2026-09-05). The slot completed nothing for three nights. `ai_job_ledger.jsonl`
  said "1 research_store_unreachable" every night from 2026-09-02, and the share
  was mounted the whole time.

  **What a sidecar's `dt` actually is.** The live SHW row
  (`b9344eb372284d7f98f6083b50178e0b`) stores `{"dt": "2026-09-01T06:30:00"}` -
  naive - while `created_at` on the same file carries `-07:00`. 06:30 is the RTH
  open on a **Pacific** desk, not on an Eastern one: the bars are DESK-local wall
  time, because the capture rail copies what the desk's own pane already drew.
  Three things followed from nobody having written that down. `_bar_moment`
  returned the naive value unchanged. `_session_close` did
  `moment.replace(hour=16)` - 16:00 in the bar's own zone, i.e. 16:00 Pacific,
  three hours past the 13:00 PT close, so the window asked for bars that do not
  exist. And `_lake_bars` handed the naive pair to `ResearchStore.read_rows`,
  whose `bar_m5.interval_start` is `timestamp[us, tz=UTC]`; Arrow does not widen
  such a comparison, it raises `ArrowInvalid`, and the blanket `except` reported
  that as an unreachable store. The same window with aware bounds returns **60
  rows**. Gate #39 could not land, and three nights of diagnosis pointed at the
  DAS.

  **The rule.** `pass_bars.desk_zone()` is the ONE named seam for the zone a
  naive desk bar stamp is written in, re-exported by `sidecar_completion` so both
  modules attach the same thing, and called through the module global so a test
  can pin it - a test that resolves the zone from the machine it runs on is only
  a test on that machine. A configured `market_local_timezone` wins, because a
  trader who has stated their zone has stated it. Failing a real IANA zone the
  platform is asked **per moment**: Windows exposes no key, so
  `market_session.get_market_local_timezone()` falls back to
  `datetime.now().astimezone().tzinfo`, a FIXED offset frozen at the instant it
  was asked, and attaching July's `-07:00` to a January bar is an hour wrong -
  twelve M5 bars of the wrong window. A naive moment is ATTACHED and an aware one
  is **never** stripped (CLAUDE.md's adoption-gate rule, applied to bar stamps).
  `_session_close` converts to `America/New_York` FIRST and then sets 16:00, so
  it is the exchange's close whatever zone the desk writes in.

  **The two absences are different and are now named separately.**
  `ResearchStore.open()` refusing is the ONLY thing that means
  `research_store_unreachable`; a read that faults returns
  `lake_read_failed: <ExceptionClass>`. That distinction is the whole cost of
  this incident: a reason that named the wrong cause sent three days of
  diagnosis in the wrong direction, and the ledger line now says which half
  broke.

  **The writer states the offset from now on** (`_serialisable_bar`), so a new
  sidecar needs no convention to be read. `sidecar_schema_version` stays **1**:
  an offset on a stamp that already had to be parsed is additive, and every old
  naive row still reads through `desk_zone()`. The original `m5_bars_ref` file is
  still never rewritten.


## Hidden is not removed (V2 item 5, 2026-09-02)

Decision 0016 answer 7 lists the surfaces the trader never opens: the Alert
Center's **Alerts**, **D1 Focus** and **Armed** tabs, and the **Universe** page.
V2 hides them behind one machine-local setting, `qt_show_unused_tabs`, default
OFF.

**They are hidden, not removed, and the next agent must not read "unused" as
"deletable".** Every one of the four is load-bearing behind the scenes:

* the **Alerts** feed is the review-alert door - `_enqueue_review_alert` routes
  through it, the M5 list is built from it, and the repetition fold writes the
  backing list before any repetition decision;
* the **D1 Focus** tab holds the flag list that `_poll_focus_d1_interest` and
  `_poll_d1_event_watches` both write into;
* the **Armed** tab is the armed-watch inventory across every symbol, and the
  expiry sweep runs at the head of the poll that owns each store;
* the **Universe** page's BUILDER writes `universe_all.txt`, which the scanner
  and now the Strength Board both read.

**How it hides.** `setTabVisible` on the existing index, never `removeTab`, so no
index shifts and nothing that remembers one - `_d1_tab_index`, `_armed_tab_index`,
`_capture_tab_index` - has to be recomputed. The left-nav page keeps its position
in `PAGE_SPECS` and its widget in `self.pages`; only the nav button's visibility
changes, so `_select_page` and every stored index keep working.

**Timers are unaffected.** Every timer behind a hidden page stays
visibility-gated exactly as snappiness packet 3 left it. Hiding costs one row of
tab strip and nothing else.

**The shortcut rule is what would actually cost the trader something.** A
`QShortcut` owned by a widget inside a hidden tab **never fires**, and two
bindings for one sequence fire **NEITHER**. `CaptureRail.action_shortcuts()` is
rebound at PANEL scope precisely so the rail's verbs survive whatever tab is on
top; a test asserts every rail shortcut is panel-scoped, that no sequence is
bound twice, and that none of them is owned inside a hidden tab.

**Hiding the tab the trader is looking at moves them to Capture** rather than
leaving them staring at a tab that vanished.

**An unreadable settings file SHOWS.** A surface the trader cannot reach is worse
than one they have to skip past, and that is the direction that cannot lose them
anything.
## P10 - one like, one dislike, from every screen (2026-09-02)

**Trader, verbatim:**

> the veto and like+claim tabs are just quicker ways to make a note for a stock.
> when I hit the dislike button in master avwap setups or not-for-today in visual
> chart review I SHOULD get a little pop-up that lets me write a note if I am not
> using the quick buttons. same if I like a stock. sometimes I may not want to
> write a note but the fact I clicked like should be processed by the bot
> eventually.

> anytime I like a D1 it should be treated with respect by the bot in regards to
> finding out what's good about it, how we can replicate those searches, and then
> how we can improve the entries. if I like a stock one day it may not be for 3-5
> days later that the best entry is.

And, decisively: **a star in Master AVWAP setups and a like in chart review are
the SAME thing.** One bucket, graded together, and the screen it came from is a
column.

### What was true before, measured on the tree that day

Three writers, one of them graded.

* **Master AVWAP ★ / ✕** wrote a review event (`favorite` / `dislike`) with
  `setup_context_fields`, plus - for the ✕ only - a `pick_feedback` row. The
  review event reaches the scoreboard and **no graded cohort at all**. So the
  most considered judgement the trader makes all day, a star on a D1 setup, left
  no forward record while the same opinion two panels away did.
* **"Not today"** wrote a `pick_feedback` verdict whose reason is the hardcoded
  string `"not today"` - never a code, never a word of the trader's own. P5
  grades it as `focus__m5_not_today`.
* **The capture rail's like** wrote a `trader_annotations` `like_claim` row,
  which `like_cohort` grades.

### The rules this produced

**One writer.** `ui/annotations/verdicts.py`. Every like and dislike from any
screen writes ONE annotation row carrying `surface` - `master_avwap_setups`,
`chart_review`, `focus_panel`, `m5_alert_bar`, `rail`. An unknown screen is
REFUSED rather than written: rows are never rewritten, so a typo would be a
permanent sixth screen no rollup knows about. These are NOT
`review_events.setup_context_fields`' `surface` values (that one writes
`"setups"`) - different file, different vocabulary, neither renamed.

**Nothing existing changed meaning.** The review event, the `pick_feedback` row
and the Focus removal all still happen exactly as they did; the annotation row is
the ADDITION, and every call site swallows its failure. An evidence store never
costs the event it records.

**The row goes first and the dialog second.** If the note box came first, Escape
would mean the click never happened - precisely the case the trader described.
The box opens only where no quick button was used: a coded dislike has already
said why in the vocabulary the scoreboard counts, and asking again would be
asking twice for one answer. The note is a SECOND row joined by `supersedes`,
never an edit.

**An uncoded veto is legal and carries no `vocab_version`.** A version stamp on a
row that cites no vocabulary would file it in a pool it was never part of -
`_rebuild_pooled_performance` pools on exactly `(vocab_version, reason_code)`. It
grades as `veto_uncoded`, never pooled with a coded cohort: a coded veto says
which of nine things was wrong, an uncoded one says only that the trader moved
on. These rows were previously SKIPPED outright by `veto_pick_rows`.

**A capture click never fetches.** The scanner-row stamp (`scan_date`,
`tracker_setup_id`, `canonical_setup_id`, `priority_bucket`, `score`,
`expected_r`) is copied from a row the desk was already showing. A bare symbol
lookup stamps nothing, because absent is a real answer and `""` would be
indistinguishable from measured-and-empty.

**A like still carries zero privileges** (plan.md P3.1). Nothing in this chain
reaches a detector, score, alert, watchlist, Focus list, review queue or
`review_policy.json`.

## R4 Part B - the four rules the code gained (2026-09-03)

Every one of these was a claim the docs already made and the code did not keep.
They are recorded here because the next agent will otherwise re-derive the wrong
answer from the shape of the code they find.

### A superseded fact pack sorts BEFORE the original, not after

`ai_jobs/setup_research._superseding` writes `<date>.json` first and appends an
ordinal on every re-run: `<date>.1.json`, `<date>.2.json`. Both Weekend Prep
readers then took `sorted(root.rglob("*.json"))[-1]`.

That is an ASCII sort. `.` is 0x2E and `1` is 0x31, so `"2026-09-01.1.json"` is
LESS than `"2026-09-01.json"` and the last name in the list is the FIRST pack
written for the day - the one every re-run superseded. Measured on the live store
on 2026-09-03, three packs existed for 2026-09-01: the original with
`gate.eligible_policy_cells = 47` in the older shape, and `.1` / `.2` with 33 in
the newer one. The reader took the original; it carries no `eligible_policies`
list at all, so `weekend_verdict.research_line` fell to its "no cell has cleared
the evidence floor yet" branch while the current pack had 33 that had.

**The rule: undo the supersession in the module that owns the naming.**
`setup_research.latest_pack_path` and `pack_sort_key` sit next to `_superseding`
so a reader can never re-derive the scheme wrongly. The ordinal is parsed as an
INTEGER - a string sort puts a tenth re-run before a ninth - and the session stem
sorts first, so a re-run of yesterday never outranks today's first pack.

**And a reader falls back rather than reporting nothing.** `eligible_policies`
arrived on 2026-09-01; every earlier pack carries the same cells under `policies`
with eligibility at `cell["stats"]["eligible"]`. On the live pack those two lists
are the SAME 33 of 73 cells, so the fallback is exact. Printing "no cell has
cleared the floor" for a pack that measured nine of them states a different fact,
and the wrong one.

### One decision graded at four horizons is one decision

`master_avwap_tier_outcomes.csv` carries one row per `(scan_row_id, horizon)` -
the tracker grades every scan row at 1, 3, 5 and 10 sessions. Reading it whole
counts one decision up to four times. `setup_docs._read_family_outcomes` did:
`avwap_band_bounce` reported n=1797 where the horizon-5 record is 329.

The rate barely moves. **The Wilson lower bound does**, and in the flattering
direction - an inflated n makes it too TIGHT - and unevenly across families,
because families are scanned at different frequencies. So it changes the ORDER,
which is the whole reason the bound is computed.

**The rule: one declared horizon, and the same one everywhere.** The value lives
in `evidence_stats.SWING_HORIZON_SESSIONS` (5) and
`autopilot_core.SWING_DIGEST_HORIZON_SESSIONS` re-exports it. R4 A11 declared it
for the AWAY digest; B2 moved the value rather than copying it, because the setup
docs answer the same question off the same file, and two horizons across the
desk's swing surfaces is the same failure as two Wilson z values. The proof it
worked: the top three families by lower bound read 0.585 / 0.543 / 0.522 on both
surfaces.

`stale_horizon == True` rows are dropped on both - "5 sessions later" indexes a
symbol's own scan rows, not exchange sessions. Only an explicit `True`; `None`
means the drift could not be measured, and uncertainty is not grounds for
deletion.

### A source-text test passes for a verb that never runs the code

The V3 item-4 guard read the TEXT of `capture_rail._record` and asserted the two
`setdefault` lines that stamp `surface` and `scan_context` were present. They
were. `commit_pass` needed the sidecar writer, so it built its own field dict and
called `record_pass_annotation` directly - it never reached that method. Every
day-trade pass on disk therefore carries no `surface` and no scan context, while
the veto, the like, the quick like and the note beside it carry both, and a rollup
by screen reads as "the trader never passes from the chart".

**The rule: assert on the row, not on the source.** `_record` gained one keyword,
`writer` - the only thing the pass path actually needed to differ on - and the
guard is five tests, one per real click handler, each performing the handler on a
rail bound to a temp file and reading the written row back. The rail under test is
told it is serving `chart_review` rather than left on the `rail` default, because
a verb that stamps nothing and a verb that stamps the default are
indistinguishable when the default is what you assert.

### A pooled cell is accumulated, never averaged

`review_preference_state.json` records what the trader took and passed per
SEGMENT and carries no side within a dimension, so a "My Decisions" row has no
direction to join `held_run_score.dimension_summaries` on - and that table was
graded in mean R alone, on the day-trade side, where decision 0016 answer 4 makes
MFE-after-a-held-level the headline.

**The rule: `held_run_score.ALL_DIRECTIONS` is a cell like any other.** It is
accumulated from the EPISODES, in the same loop, and summarised by the same
`Segment.summary`. It is never the long cell averaged with the short cell: a mean
of trimmed means is not a trimmed mean, and computing one in the panel would be
the second formula that R4 A10 deleted, returning under a different name.

The same section is why the tracker's other two columns were labelled. The
champion tier (PROVEN / MUTED / active) says whether the desk should ALERT on a
segment at all; `Verdict` is the aggregator's `edge_score`, computed from average
R; Held x Ran is what the alert offered once the level held. Three questions. Two
of them sat unlabelled next to each other, and a reader with one number and two
meanings will pick the flattering one.

**A segment the learning state has never seen is BLANK, not "active".** "Not
tracked" and "tracked and unremarkable" are different facts - live, 104 of 295
rows are the first and 185 the second.


## F1 - the desk freeze of 2026-09-03: a build thread that owned the GIL

The trader at ~09:00 PT: *"the program has been freezing and has been basically
unusable all morning"* ... *"fix it"*. This is what was under it, measured on the
running desk (pid 11612, on the old `main` tip `93732ef`) rather than reasoned
about, and the three rules that came out of it.

### What was measured

- `uvx py-spy record --gil` on pid 11612, 08:45-08:55 PT: the **`qt-warehouse-build`
  thread held the GIL in 82.7% of samples**; `MainThread` got **2.3%**. From
  outside the process, WM_NULL pings to the desk window measured **100-606 ms**
  hangs every few seconds. That is the freeze, exactly: the GUI thread was not
  slow, it was not scheduled.
- **84% of that thread's samples were inside
  `scripts/research_warehouse/exchange_calendar.py`** - `session_for` ->
  `trading_session` -> `is_trading_day` -> `holidays(year)` - recomputing Easter
  and five nth-weekday walks once per M5 bar per occurrence, with nothing cached.
  Benchmarked in the desk venv: 20,000 `session_for` calls, **0.25 s uncached,
  0.0114 s memoized (21x)**.
- `research_lake/manifest_log.jsonl`: the `m5_close_recipe_outcomes` stage ran
  **27-57 minutes after EVERY scan** (09-01: 28/51/57 min; 09-02: 27/38/44; the
  09-03 build started 07:59 and was still running at 08:55). One build per scan,
  **four scans a day, all inside RTH**.
- `ui_stalls.jsonl` **stopped at 06:03:35** because `MAX_RECORDS_PER_SESSION =
  2000` had been spent overnight: the desk came up at 21:04 the night before and
  wrote **1,614 records between midnight and 06:03**, the 04h and 05h hours
  burning ~500 each on sub-second native `app.exec` stalls on an idle desk. So the
  morning the trader called unusable has **no stall evidence at all** - the one
  morning the diagnostic existed for. A per-DAY cap of 2000 would have gone blind
  at the same minute.

### The rules this produced

**The post-scan warehouse build runs in a CHILD PROCESS at below-normal priority,
never a thread.** A CPU-bound Python thread holds the GIL by construction: there
is no priority setting, no timer, no chunk size and no `sleep(0)` sprinkle that
gives the GUI thread back, and every one of those would have been a plausible
"fix" that measured nothing. LD-01 specified this work as a *post-scan/EOD CLI
build job* in the first place; running it in-process was the deviation.
`ScanService.start_warehouse_build` now spawns
`research_warehouse.cli build --run-id <id>` (frozen: the app's own
`--warehouse-build` flag, because a frozen `sys.executable` is `TradingBotV3.exe`
and parses `-m` as its own CLI - the same trap that silently killed every
scheduled scan from 2026-08-12), with `BELOW_NORMAL_PRIORITY_CLASS |
CREATE_NO_WINDOW` read by name through `getattr` so macOS still launches. The
child is registered with `_register_owned_process`, so shutdown reaps it - and a
reaped build is safe because the build's `single_flight` lock **reclaims a dead
holder rather than obeying it**. One daemon thread, `qt-warehouse-build-wait`,
blocks on the child's pipe; blocking on I/O holds no GIL, which is the entire
distinction this rule rests on. Detail and the reopen triggers: BD-95.

**A build child is owned, and is not a scan child.** `ScanService._start` refuses
a new scan while `owned_scan_process_count()` is non-zero - "previous scan child
still running" - so registering the build there and stopping would have converted
this freeze into a different failure: a 27-57 minute build, four times a session,
refusing the next scheduled scan. The build is registered for the shutdown reap
and appears in `owned_scan_process_snapshot`, which is the reaping account; it is
excluded from `owned_scan_process_count`, which is the may-a-scan-start question,
and `owned_build_process_count()` answers for it. Two tests hold this: one on the
counts, one driving the real refusal path.

**The exchange calendar is memoized.** `holidays(year)`, `half_days(year)` and the
session builder behind `trading_session` are `functools.lru_cache(maxsize=None)`.
The cache sits behind `trading_session` in a positional
`_trading_session(day, calendar)` rather than on the public keyword-only
signature, because `lru_cache` keys on the **call shape**: `trading_session(day)`
and `trading_session(day, calendar="XNYS")` are the same question, and decorated
directly the answer is built and stored twice - which is how the first version of
the identity test on `session_for` failed. `TradingSession` is a frozen dataclass,
so sharing one instance is safe. The holiday dicts are **shared and must never be
mutated**; every caller in `scripts/` and `tests/` only reads them (checked
2026-09-03), and a caller that ever needs to mutate copies at its own call site.

**The stall watchdog's record cap is per HOUR, not per session.** `_write` keeps
the session total (`records_written` is unchanged, and `session_summary` reads
what it always read) and gates on a separate `_hour_records` reset whenever the
local `%Y-%m-%d %H` key changes. A runaway loop is still bounded - 48k records a
day, ~50 MB - and a quiet night can no longer spend the trading morning's budget.
The general form of the lesson: **a diagnostic's budget must roll on a window
shorter than the thing it is meant to observe**, or the desk goes blind precisely
when it has been up long enough for something to be wrong.

## One chart on the Trading Desk (2026-09-03)

**The trader, verbatim:** *"when i click on a ticker anywhere while on the trading
desk tab, i want the chart to come up on the visual chart review chart we have in
the center of that tab. right now i click things in the auto RS/RW board or the
master avwap setups board and it does a pop up. thats fine on other tabs, but the
main tab should always be centralized with the main chart"*

**What was true before.** The M5 Strength Board had charted into the pane since
2026-08-31 (its entry is above, under the Strength Board rule). Every other click
surface still opened `show_symbol_snapshot`: the RS/RW, entry and Focus-strength
boards in the Alert Center, the feed's ticker-name label, and the four setups-column
panels (setups table, RS Window, Industry Board, Watchlists). Each was right when it
was written - the popup was the only chart a board on another page could reach -
and became a window in the way once the boards and the pane shared one screen.

**The rules.**

- A board INSIDE the Alert Center always charts in the pane, through
  `_chart_board_symbol` = `chart_symbol` with a named origin. It is in the same
  column as the pane in every mode, so there is no case for the popup.
- A feed ticker-name click is the same as a row click (`_show_alert_detail`): the
  real alert, with its trigger. A `MANUAL_CHART` of the same name would have thrown
  away what the scanner said.
- A setups-column panel is a column of the desk in workspace mode and a sub-tab of
  its own in tabs mode. It carries `set_chart_sink`; the desk sets it in workspace
  mode and clears it in tabs mode, because a chart drawn on a sub-tab the trader is
  not looking at is worse than a popup. `None` - the default - keeps the popup, so
  a standalone panel and every test of one behave as before.
- The popup is not retired. `show_board_symbol` is still the AWAY Recap's door, and
  a page that is not the desk keeps it.
- The click still uses the lookup box's door and never `_enqueue_review_alert`, for
  the four reasons in the Strength Board entry.

## T1 - the capture window is the why, and a look is not a queue (2026-09-04)

**The trader, verbatim:**

> when i double tap something in the capture window (either veto or like+claim) i
> shouldnt get a pop up note box. the point of the capture window is to quickly
> enter "WHY" I like or dislike something. Additionally the "like" button in the
> visual chart review should NOT advance the char to the next page because i still
> need time to enter alerts etc. not today can continue to go to the next chart
> with a pop up note box.
>
> I want all shorts and longs on the RS/RW board TC2000 to bne auto added to the M5
> focus picks. additionally when I click on ANYTHING from the RS/RW board it should
> not make a queue of picks if I click on more nor should it add to the "waiting"
> list. once i look and click off, its done.

### What was measured, on `main` @ `6e05878`

- **One veto click wrote TWO veto rows and opened a box.** The rail's `commit_veto`
  wrote the CODED row and emitted `captured(EVENT_VETO)`; the pane forwarded it as
  `removeTodayRequested`, which is the "✕ Not today" BUTTON's signal; the panel's
  `_remove_review_alert_for_today` then called `_record_not_today_annotation`, which
  wrote a SECOND, UNCODED veto row through `verdicts.record_not_today` and opened
  `open_note_prompt`. The box asked for a why the trader had just typed.
- **The same was true of "Veto D1 - but M5 today."** `_veto_but_day_trade` ended in
  the same method, so the day-trade veto wrote the uncoded row and opened the box
  as well. The packet had called that verb untouched; the lead ruled on 2026-09-04
  that the trader's "either veto or like+claim ... no pop up note box" covers it.
- **A like took the chart away.** Every like path - the claimed like, Alt+L, and the
  chart's "♥ Like" button - reached `_record_like`, and `_advance_after_like`
  called `_advance_review_queue`. The trader was arming alerts on a chart that was
  already gone.
- **Five board clicks built a four-deep waiting list.** Every board in the alert
  column charts through `chart_symbol`, which stamps `MANUAL_CHART_TAG`.
  `_select_review_alert` set `_current_review_holds_place = not
  _is_m5_review_alert(alert)`, and that method returns False for `MANUAL_CHART_TAG`
  - so a look HELD A PLACE and the next board click pushed it to the head of the
  queue. Clicking META, NVDA, AMD, SOXL and TSLA left `['SOXL', 'AMD', 'NVDA',
  'META']` waiting and the pane reading "4 waiting".
- **The TC2000 board reached Focus only by hand**, through `_add_symbols` ->
  `focus_service.add`, which is a trader LIKE and writes a `pick_feedback` row.

### The rules this produced

- **A VETO retires the chart; a LIKE and a NOTE never do.** The rail's veto has its
  own verb, `vetoRetireRequested` -> `_retire_after_veto`, and writes ONE row and no
  box. `removeTodayRequested` is the "✕ Not today" BUTTON's signal alone, and that
  button is unchanged - uncoded row, box, advance - because the trader kept it in so
  many words. **The day-trade veto retires through the box-free verb too**, after its
  Focus placement, in that order, and a failed placement still retires.
- **Both retirements are ONE body with a flag** (`_retire_review_alert(...,
  write_not_today_annotation=)`). The auto-pick, faded and Focus-review branches each
  return early; a second copy of that ladder would have started parking symbols that
  must not be parked.
- **A like is a REPORT, not a request.** `likeRecorded` -> `_after_like`: the review
  event, a status line, and nothing else. **Its event is still named `like_advance`**
  - `review_learning.TAKE_ACTIONS` keys on the exact string, and renaming it would
  drop every past like out of the take side of the scoreboard. The name is historical
  and now means "liked; the symbol keeps alerting and the chart stays". **Since the
  second pass below this is the QUICK like only.**

- **A look is not a shown alert.** A `MANUAL_CHART_TAG` chart holds no place, and
  clicking away from one writes NOTHING - not a re-queue and not a `skip`, because a
  look belongs in no P(take | shown) denominator. `_is_manual_chart_look` is a
  separate exact test rather than a fold into `_is_m5_review_alert`: that method
  answers "is this a LINE IN THE M5 BAR", and the two questions share a tag but not
  an answer. The M5-alert-bar `skip` with `clicked_away_from_m5_alert` is a different
  population and is byte-for-byte untouched; a dequeued D1 chart still returns to the
  head of the queue.
- **The TC2000 board's parity rows auto-join M5 Focus**, on `boardChanged` and once
  at attach. Only rows with an EMPTY `failed_floors` - a greyed near-miss is a name
  that missed one of the trader's own filters. The ONE adoption gate is re-run on
  each row's own numbers (the board can be fifteen minutes old) and UNKNOWN fails. A
  symbol in `_ignored_symbols` is skipped, so the next refresh cannot undo a "Not
  today". DESK only; the auto-mode matrix is unchanged.
- **The machine writes through the STORE, never `FocusService.add`** - the same
  reason the regime-pause auto-join does. `store.add` then `mark_auto_adopted`, and
  the marker only when `add` actually added: an existing unmarked entry is the
  trader's and must not change owner. It **never removes**; the ten-session fade and
  "Not today" own removal.
- **A look at a name that was WAITING takes it out of the waiting list**, and it
  does not come back. `_select_review_alert` drops both the outgoing and the
  incoming symbol from the queue before it decides what to do with the outgoing
  chart, so charting a queued name from a board and then clicking away leaves it
  out: the trader has now seen it. That IS *"once i look and click off, its
  done"* and it is deliberate; the chart the look REPLACED still returns to the
  head if it held a place, and no `skip` is written for the look either way.
- **The board must not undo a removal, and `_ignored_symbols` was not enough**
  (fix round 1, 2026-09-04). That set only ever holds names the "Not today" verb
  parked. FOUR other doors remove a Focus pick without parking anything - the
  Focus-review walkthrough (`FOCUS_REVIEW_TAG` → `remove_everywhere`), the Focus
  list's own remove button (`focus_picks_panel._remove`), the chart's cross-focus
  toggle (`toggle_m5_focus`) and the Master AVWAP unfavorite - and the next
  fifteen-minute refresh put every one of them straight back, **re-injecting the
  name into `longs.txt` with it**. Reproduced end to end: adopt NVDA, remove it
  through the Focus-review walkthrough, republish the same board, NVDA is back.
  The record is kept in the **STORE**, not at each door: `FocusPickStore` writes
  a `(symbol, side, category, session_date)` row under an ADDITIVE `declined` key
  in `focus_auto_picks.json` on every removal (`remove`, `remove_everywhere`,
  `clear`, the fade; `remove_if_auto_adopted` delegates to `remove`), and
  `declined_today` answers for TODAY only - a new session clears the meaning and
  `_load_declined` prunes older rows so the file cannot grow. Deliberately not
  conditional on a marker existing: a name the trader typed and then deleted is
  exactly the name the machine must not put back. Adding the name back by hand
  clears the decline, because that is the trader changing their mind, and
  `expire_m5_if_new_day` clears the declines with the markers on the day roll.
- **Every adopted name is also injected into the shared `longs.txt` /
  `shorts.txt`.** That is `FocusPickStore._inject_into_shared` and it is
  pre-existing behaviour of every Focus add, but it is worth saying out loud
  here because the auto-join is the first path that adds names WITHOUT the
  trader clicking: it grows BounceBot's intraday scan input. Measured on the
  live store the day this landed: `longs.txt` 29 names, `shorts.txt` 50, of
  which 33 + 32 were store-injected m5 entries. A removal un-injects again,
  which is the other half of why the decline record has to exist.
- **Adds are BATCHED, one `add_many` per side.** `add_many` rewrites the focus
  file, the membership file and the pick clocks once for the batch; sixty names
  through `add` measured 781 ms on the Qt thread, and nothing this panel
  controls bounds the board's row count. The MARKER stays per name - it carries
  that row's own strength.
- **The review event counts who already owned each name** (`already_auto` /
  `already_trader_owned`, read through `store.is_auto_adopted` exactly as the
  regime-pause auto-join does). Counts only: a marker is never written over a
  name the trader typed.
- **One review event per refresh that adopted or refused anything**,
  `strength_board_auto_focus`, carrying `side_counts`, `adopted`, `refused` and
  `as_of`. `record_review_event` refuses a row with an empty symbol, so this one
  carries `symbol="M5_STRENGTH_BOARD"` - the event is about the BOARD, the names are
  in the detail, and an underscore makes that value unrepresentable as a ticker under
  `ui.models.bounce.SYMBOL_RE`, so no symbol-keyed join can ever match it. No scanner
  alert was invented for it.

#### Second pass, 2026-09-04 (packet T2): the claimed like is one double-click

The trader read the T1 tree on the desk and answered:

> pretty close. for the "like and claim" part of the capture tab, a double click of
> any of the setups there should be sufficient. I shouldnt have to type anything
> below that box. and then double clicking that box should advance the chart.

So the two like modes part company, and R9.2(a)'s required why is superseded for the
CLAIMED path as well as the quick one:

- **A claimed like needs no why.** `commit_like` refused an empty one and refocused
  the field (`_prompt_for_why`, now deleted). It records whatever is in the field,
  empty included; a whitespace-only why strips to nothing and the row simply carries
  no `note`. The claim itself is the label a later reader can check, which is what
  the 2026-08-22 `dislike_reason` failure lacked - the why was the ONLY label there.
  The trader's own prose is still worth more than anything the machine derives, so
  the field stays, relabelled "why (optional)".
- **A claimed like ADVANCES; a quick like still does not.** `_on_captured` reads the
  row's mode through `like_mode_of` (absence reads as claimed, the P9 rule) and fires
  `likeAdvanceRequested` or `likeRecorded`. Two signals rather than one with a flag:
  the host's two answers really are different verbs, and a flag is read wrong once.
- **An advance is NOT a retirement.** `_advance_after_like` records and calls
  `_advance_review_queue`, and nothing else: `_ignored_symbols` untouched, so the name
  keeps alerting and keeps reaching the hourly D1 phone push; no auto-adopted Focus
  pick dropped; the symbol's other queued alerts keep their places; nothing placed.
  That is R9.2(b)'s measured harm - 40 of 52 likes parking their own symbol - staying
  fixed while the movement comes back.
- **One recorder, two callers.** `_record_like_advance` writes the `like_advance`
  event for both handlers. Two copies is exactly how the quick and claimed paths
  would drift, one gaining a field the other never got.

## The research tee burned a core (2026-09-03 evening)

### What was measured

The desk was restarted at 13:02 PT onto `f903ca4` (past F1). At 21:05 PT, five
hours after the close, `python.exe` was at **101% of one core**: 29,909 CPU-seconds
in eight hours, **26,540 of them on one thread, `warehouse-m5-tee`**. A 15-second
`uvx py-spy record --gil` put **330 of 362 GIL-holding samples (91%)** in
`research_warehouse/bar_archive.py::capture_m5_tee`; the GUI thread appeared in
**0 of 362**. The largest leaf was `session_context` ->
`_market_session_module` -> `_ensure_scripts_on_path` -> `Path(__file__).resolve()`,
a real-path syscall made once per cached bar (197 us, benchmarked in the desk venv),
then `get_market_session_window`, then the sha256 in `_source_hash`.

The mechanism: `bounce_service.capture_warehouse_tee` fires every 60 s and hands
the tee a copy of the whole `bot.latest_bars` - 888 symbols x 5 sessions x 78 bars =
**346,111 bars** after the close, 275 symbols right after a restart. The tee parsed,
session-tagged and hashed every one of them and only THEN checked the `seen` set.
That is at least 72 s of work per walk against a 60 s timer, so the thread never
rested. The 2026-09-03 timer recon had ranked this timer "cheap, low confidence"
from its docstring; nobody had sampled the GIL after F1 moved the build out.

The same day's stall log (`ui_stalls.jsonl`) carried 5,719 records and 1,336 s
blocked, **816 s of it attributed to `app.py:1234`** - the event loop itself -
because a stall caused by another thread holding the lock leaves the GUI thread's
own stack innocent. The first M5 scan cycle after the restart logged a **1,751 s
preamble** (`focus_fast_lane` 1,210 s); the old desk's RTH preambles the same
afternoon were 513-535 s against a 300 s candle.

The second half: `_session_seen` keyed its set on `moment.date()` of a UTC
moment. At 00:00 UTC (17:00 PT) the set emptied and the tee re-spooled the whole
five-day cache: `segment-20260904T000029-*.open.jsonl`, **346,111 rows / 240 MB**,
four of its five sessions already in the lake. A restart did the same (107,119
rows at 13:05). The seal published whatever the spool held, so
**`bar_m5 month=2026-08` held 12,015,283 rows for 1,816,970 distinct grain keys
(85% duplicates)** and `month=2026-09` 541,444 for 208,841. The derived bars and
intraday features for those months were computed from the duplicated rows.

### The rules this produced

- **The tee de-duplicates BEFORE it does any per-bar work, and its mark is
  persisted and never reset by a clock** (BD-96). `capture_m5_tee` runs two
  passes: identity first (timestamp, forming check, high-water / `seen`), then
  prices, hash and session tag for survivors only. The live desk's state is a
  per-symbol high-water mark in `tee_high_water.json` beside the spool; a symbol
  whose newest bar is behind its mark is not walked. A restart resumes; a UTC
  midnight changes nothing.
- **The seal de-duplicates at the dataset grain and counts what it drops.**
  Trusting an upstream dedupe was the defect. `SealResult.rows_deduplicated` is
  the number; superseding datasets are exempt.
- **A repeated grain key in the lake is repaired by a COMPACT-shaped rewrite,
  never by deleting files**: `research_warehouse.cli dedupe --apply`, dry run
  by default, earliest observation kept, inputs retired, the drop written on the
  manifest line.
- **Every desk thread's CPU time is measured once a minute and a hot one is
  named** (`ui/thread_cpu_gauge.py`, always on, `thread_cpu.jsonl`). The stall
  watchdog can only name a stall the GUI thread caused; this gauge names the
  thread that starved it. On 2026-09-03 that answer took a py-spy session at
  21:05 for a thread that had been hot since 13:02.
- **A recon that rates a timer from its docstring has not measured it.** The F1
  packet fixed the build thread; the tee thread had the identical shape and was
  found the same evening by sampling the GIL, not by reading the code.

## Q4 - the overnight run protects its deterministic work (2026-09-04)

Three rules `CLAUDE.md` gained. The long form of each lives in its own governing
document rather than being retold here; this entry says which, and states the one
thing a future editor most needs and would otherwise have to rediscover.

**The slot order is decision 0018's, and the reason is a reservation, not a
preference.** Full record:
[`docs/decisions/0018-deterministic-stage-before-narration.md`](decisions/0018-deterministic-stage-before-narration.md).
The thing to know before touching `default_slots()`: the runner does not queue or
shorten a slot whose reserve no longer fits the remaining window - it records
**SKIPPED**, and the night simply does not do that work. `ai_summary` (up to ~170 min
in chunked mode) and `ticker_briefs` (120 min) sat ahead of every deterministic slot,
and the 2026-09-01 run took six hours. A skipped narration is regenerable tomorrow; a
skipped cohort grade, sidecar completion or fact pack is a hole in an append-only
forward record for a session that is over. Verified at the code level before the move:
**no deterministic slot reads either narration slot's output file** - `daily_digest`
imports `ai_summary` as a LIBRARY to narrate its own pack and opens neither slot's
published file.

**The digest gate's two halves.** Full record:
[`docs/LOCAL_AI_AUTOMATION_PLAN.md`](LOCAL_AI_AUTOMATION_PLAN.md) §7.0. Two things a
future editor will want and would otherwise guess at:

* **"Clean" reads the `unavailable` map, and that is the whole failure record the pack
  has.** There is no `failures` key, no `errors` key and no coverage-failed flag; the
  pack's own `summary` already renders a non-empty `unavailable` as "this pack is
  INCOMPLETE rather than empty". A pack written with no AI store configured records
  `ai job ledger: No AI store configured` and is therefore NOT clean - which is why two
  existing test helpers patch that read rather than assert a dirty pack is clean.
* **The approval file is deliberately outside every automatic path.** A nightly job that
  could write `digest_audit_approval.json` would turn "the trader audited three packs"
  into "the runner asserted it did", which is precisely the claim the gate exists to
  make impossible. A test walks `runner.py`'s source and `run_daily_digest`'s AST to
  keep it that way.

**`entry_index.json` fills nothing its sources do not establish.** Two of its four
sections - `swing_win_rates` and `journal_execution` - are empty with the reason
printed in the file, because the daily fact pack carries champion INTRADAY outcomes and
no journal block. The temptation is to fill them from `master_avwap_tier_outcomes.csv`
and the journal store; that would put two grains in one index and let a reader compare
them. A blank is right where the question cannot be asked of this record.

**A citation names the file the numbers came from.** Reviewer blocker, caught before
merge: the index cited `facts_path(root, day)` - always version 1 - while every value was
read from `latest_pack_files_by_session`'s newest sibling. **Three of the nine live
sessions are superseded** (`2026-08-25.2.json`, `2026-08-26.2.json`, `2026-08-27.3.json`),
so on a third of the store the index handed the reader the pack that had been corrected.
`read_fact_pack_files` exists to carry the path beside the payload. The tie between two
siblings with the same `generated_at` breaks on the SUPERSESSION INDEX, never on the file
name: `2026-08-25.1.json` sorts before `2026-08-25.json` alphabetically, which would give
the correction's place to the pack it corrects.

**A ratio and a verdict must come from one number.** Same review: both gate counters
passed `have=sessions_collected` - the distinct count Q4.1 deliberately kept for pre-Q4
readers - beside a `met` that turns on the consecutive run. Ten scattered packs and a
two-session run rendered "Digest 10/10" and not met. `gate_counters._digest_have` is the
one place that answers it, and the strip test asserts the TEXT rather than the flag,
because the text is what the trader reads.

**`.git` is a FILE in a git worktree.** `definitions_git_commit` reads `.git/HEAD`
directly and therefore returned `""` for every index built by an agent - which is all of
them. `digest.repo_commit` follows the `gitdir:` pointer, then `commondir` for the refs
and `packed-refs`, and still yields `""` rather than failing: provenance is evidence, not
a gate.

---

## N2 - the synthesis was not malformed, it was cut (2026-09-05)

Two of the last four nightly `ai_summary_*.json` files carried
`map_reduce.synthesized: false`. Both errors read as a broken model:

```
RuntimeError: local provider returned invalid summary JSON after 2 attempt(s):
  Unterminated string starting at: line 1 column 14502 (char 14501)   <- run of 2026-09-03 02:10
  Unterminated string starting at: line 1 column 14709 (char 14708)   <- run of 2026-09-05 02:41
```

Both offsets are **3,500 tokens of dense JSON at ~4.2 characters per token**, and 3,500
was the single hard-coded `max_tokens` that every local call sent - the map slices, which
answer with a handful of findings about one chunk, and the reduce call, which answers with
the whole document. The parser's complaint was true and pointed at the wrong thing: a
length cut is a valid JSON PREFIX that stops, so it always reads as an unterminated
string. Worse, the existing retry re-sent the **identical** request with the validator's
rejection appended - more prompt against the same ceiling - so it cut again in the same
place, at roughly seven minutes of generation per attempt. `unsynthesized_summary` then
published 12 rows per section and hid 52 and 39; it is designed as a rare last resort and
it fired on half the nights.

There are now **two output caps**: `LOCAL_MAP_GENERATION_TOKENS` (3,500) and
`LOCAL_SYNTHESIS_GENERATION_TOKENS` (8,000), selected per request by
`local_generation_tokens(evidence)` from the `map_reduce_synthesis` scope that
`findings_package` already stamps on the reduce package. `LOCAL_GENERATION_TOKENS` stays
as an alias of the map cap because the evidence budget and two test modules import it.
**The evidence budget keeps subtracting the MAP cap**: the reduce prompt is the model's
own findings, a few tens of KB against a 65,536-token window, so 8k of output sits beside
it - and widening what the synthesis may WRITE must never narrow what a map slice may
READ. The cloud payloads are untouched at 3,500. A length stop is now read from
`choices[0].finish_reason` or Ollama's top-level `done_reason` **before the text is
parsed**, and earns ONE retry asking for at most 8 findings per section; a second cut
raises `LocalOutputLengthError`, whose `stop_reason` reaches the manifest as
`synthesis_stop_reason` instead of being re-derived from a message.

**The map-slice failures are a DIFFERENT bug and were not fixed by this** (packet item 0).
The desk log names each one, and the 2026-09-05 pair are Q3 grounding rejections:

```
map slice 18/47 (setups.short_horizon [1/1]) failed: ... 1 row(s) dropped:
    what_is_working[0]: numeric claim without a resolvable metric_ref
map slice 32/47 (setups.playbooks [3/11]) failed: ... executive_summary cannot be blank
```

Every earlier one in the log (2026-08-28 through 2026-09-04) is the same family - "every
citing statement was unsupported". Those are exactly what the rejection-feedback retry is
for, and they are unaffected by an output cap. The map half of the length detection is
therefore **unobserved live** and is covered by test only.

---

## Overnight AI repair - import seams and grammar fallback (2026-09-17)

`theta_pick_tracker` imports a focused `master_avwap_lib` module, whose compatibility
package loads `legacy` and then `runner`. A top-level runner import of the tracker recorder
therefore requested `record_theta_picks` from a partially initialized module. The runner now
has a module-level lazy forwarding function: the scan resolves the real recorder only when it
writes its existing shadow evidence, while callers retain the `runner.record_theta_picks`
monkeypatch seam. No theta score, row identity, recorder timing, stage order, or evidence
semantics changed.

The local backend has a second narrow compatibility defect: it rejects the full enrichment
schema only while compiling grammar when `summary.maxLength` is exactly 2,000. The first
request retains that full contract. Only an explicit HTTP 400 grammar parse/initialization
message earns one `json_object` fallback; the returned object is still checked against the
original closed schema, including the 2,000-character limit. Other HTTP failures never retry.
The repair changes neither model nor timeout, never changes journal trade selection, and leaves
prior artifacts intact on a failure.

## Night kinds - the night picks the slate, and nothing runs by day (2026-09-19, TJ-13A)

The long form behind the CLAUDE.md rule *"Local inference is night-only, seven days a
week, and the NIGHT picks the slate."* Trader, 2026-09-19: *"I keep the computer on
overnight in the weekends too. … I always want the bot to run overnight never during the
day so I can restart it or use it for market prep."* Decision 0021 answer 19; plan.md
§12.4 TJ-13 items 5, 6 and 8; decision 0018's 2026-09-19 amendment. Branch
`claude/tj13a-night-slates`, merged `9eaae1dd`.

**What was measured.** The `TradingBotV3 AI Jobs` task already fires 22:00-06:00 Pacific
seven nights a week as its own process, but `window.is_weekend` short-circuited the
configured clock, so the whole of Saturday and Sunday counted as inside the window - on
the two days the trader is at the desk all afternoon. The weekend branch of
`window_close_at` then walked forward to the next WEEKDAY morning, so a job starting
23:00 Pacific Saturday was told it had until 09:00 ET Monday: 31 hours, and every
`reserve_minutes` check passed on a number that was never true. The honest answer is 420
minutes. `weekly_synthesis` had never run in 476 ledger rows because it needed a typed
command, and `ai_summary` ran 12,453-18,540 s a night, ending `degraded_no_narrative`
four nights running.

**The rules.**
* The configured ET window applies EVERY day. The market-session block keeps its own
  weekend short-circuit - that is the sec 2 hard rule's fail-closed design, it must not
  need the calendar on a day the exchange never opens, and TJ-13A did not touch it.
* `--force` buys the attempt caps and the already-completed check. It does NOT buy the
  clock for a `JobSlot.uses_model` slot. A deterministic slot is seconds of work and no
  model, and stays forceable by day - that is the repair the flag exists for.
* `uses_model` is DECLARED per slot, never inferred from a name, so a later packet that
  adds a model to a slot says so in the same edit. `daily_digest` carries it: its facts
  are deterministic, its second artifact is narrated.
* A night is keyed to the EVENING IT STARTED. Noon ET is the split, which sits outside
  both the live 01:00-09:00 ET window and the 18:30-08:00 default, so an evening firing
  keeps its own date and a small-hours one belongs to the day before. In ET, Saturday
  night's firings are already stamped Sunday - that is exactly the seam a date-only
  reading gets wrong.
* The kind comes from the exchange calendar, not the weekday number, so a Monday holiday
  moves the Sunday slate to Monday night and a Friday holiday starts the weekend a night
  early. In a Friday-holiday week Friday AND Saturday night are both `saturday`, and the
  second is the resume night.
* `night_kind` raises when the calendar cannot answer; `run_ai_jobs.py` catches it and
  uses the WEEKNIGHT slate - the light one - because "we could not tell" must spend the
  least, and `run_slots` still refuses outright a moment later rather than keying
  artifacts to a guessed session date.
* `EXPECTED_SLOT_ORDER` is the order WITHIN a night. Every slate is a subsequence of it.

**A typed `--slot` is the operator's explicit choice.** It resolves against every
registered slot - `default_slots() + optional_slots()` - and not against tonight's slate,
because a slate is what the night does UNATTENDED. Filtering a typed name by the slate
made `--slot ai_summary` on a weeknight a silent no-op that exited 0, while
`run_ai_jobs.py`'s own docstring advertised that exact command; "I typed it wrong" and
"it ran and found nothing" must not look alike, so an unknown name is an error that lists
the valid ones. This widens what can be NAMED and never what can RUN: a model slot named
by day is still refused by the night-only window, and its ledger row still says so.

**The page button is gated too, and it was already shut.** `AiSummaryPanel
.generate_summary` now asks `window.launch_allowed()` before a LOCAL run and refuses with
the window's own reason ("Local AI runs at night only, seven days a week: …"), starting
no thread. There is ONE window in the system, not a second one on the panel that could
disagree. A CLOUD provider is untouched: the rule is about this desk's hardware, and a
metered API call competes with nothing here.

RECORDED rather than repaired: that door was already shut for a different reason.
`AiCredentialVault` knows `openai` and `anthropic` only (`ai_credentials
.PROVIDER_ENV_KEYS`), so the panel's local path has always stopped at "unsupported AI
provider: local" before any thread was created - even though the combo box offers "Local
(on this desk)" whenever `ai_local_endpoint_url` is set. The window gate is therefore
defence in depth, and enabling the button to actually reach a local model is a
capability, which is the trader's call and not this packet's.
`tests/test_tj13a_panel_night_only.py` pins both facts, so whoever enables it finds the
gate already in front of them.

**A forced daytime run may still do a slot's DETERMINISTIC half.**
`JobSlot.model_free_kwargs` declares the keyword arguments that make a slot's own `run`
model-free. Only `daily_digest` has any - `{"narrate": False}`, a switch
`run_daily_digest` carried before this packet - because its fact pack is deterministic
and its narration is a second artifact that already degrades on its own. Marking the slot
`uses_model` had taken the fact pack away from a forced daytime run; this gives it back
without reopening the door. **The forced daytime run writes a superseding facts pack and
never overwrites last night's narrated digest**, and the ledger row says what was left
out, so it can never be read as the night's full digest. It is KEYWORDS rather than a
second callable because the house test pattern is `dataclasses.replace(slot, run=<spy>)`,
which would not replace a second callable and would reach the real job through it. A slot
with no model-free half declares None and is still skipped - there is no summary without
a model.

## The synthesis call is the one nobody budgeted (2026-09-19, TJ-13A)

TJ-13A item 3. Two defects sat in one ledger line, 2026-09-17: *"53 of 53 slice(s);
completion=unsynthesized_fallback after RuntimeError: local AI endpoint … is unreachable:
… Read timed out. (read timeout=900)"*.

1. **A dead endpoint cost the whole night.** Every slice was attempted in turn and each
   had to reach its own 900 s timeout, so an endpoint not answering at 22:00 was
   discovered at 03:00. `ai_summary.LocalEndpointUnreachable` and
   `is_endpoint_unreachable` name that failure - by class first, by the sentence second,
   because a caller that wraps or re-raises it loses the type and keeps the words.
   `run_map_reduce` gives up on it at once. It keys on UNREACHABLE and NEVER on any
   failure at all: an ordinary bad slice still costs its own slice. The 30-minute re-fire
   is the retry ladder for a transient blip.

   **Say "on the first call", not "in seconds".** A REFUSED endpoint - nothing listening -
   is measured at 2.06 s, and the slot degrades about that fast. A HUNG one still costs
   ONE full 900 s read timeout on slice 1, because the timeout is what discovers it; what
   the repair removes is the other 52. The difference is worth writing down so nobody
   reads a 15-minute degrade as a regression.
2. **Every map slice is chunked to fit; the reduce call was not.** The package that has
   to hold the whole night was 119,677 characters against the local evidence budget -
   **78,119 characters as this desk is configured today**, derived from the 64k context -
   which is how a 900 s read timeout became the normal ending on 09-15, 09-16, 09-17 and
   09-18. The bound is that setting, read at run time, so it tracks the context rather
   than a number written down twice. `bounded_findings_package` binary-searches the
   largest HIGHEST-CONFIDENCE prefix that fits `evidence_budget_for("local",
   tier="medium")` - the existing setting, not a new number - rebuilding the package for
   each measurement rather than estimating, because the skeleton, the aliases and the
   hash all cost characters.

**Bounding is not silent truncation.** `map_reduce.findings_dropped_to_fit` is ALWAYS
PRESENT - 0 on a night where everything fit, so "0" cannot be read as "this build did not
look" - and the coverage statement says the number in words, the way it already names a
failed slice.

**A rejected model reply is now kept.** `ai_summary.record_rejected_reply` writes one
bounded JSON file per rejection under `project_paths.AI_REJECTED_REPLIES_DIR` - **LOCAL**,
under the runtime tree, and never the AI store, which is the DAS and can be asleep: this
runs inside a slot, and a ~20 s spin-up to file a diagnostic would make the record cost
more than the thing it records. No DAS path is reachable from the slot. Temp-and-rename,
bounded twice (`MAX_REJECTED_REPLY_CHARS` 20,000 chars per file,
`MAX_REJECTED_REPLY_FILES` 200 files kept), pruned after the write, and it NEVER raises
into the slot: an evidence store is never allowed to cost the thing it records, and this
one records a failure the ledger is already reporting properly. It exists because three
identical nights of `journal_enrichment` failure left only the validator's sentence, and
the shape behind it had to be reconstructed by hand.

**`ai_summary` is capped at three attempts per session, and runs for ONE session a week**
(Friday's, on Saturday night - Monday through Thursday sessions get no AI summary).
`max_attempts=0` was written when a failing summary was expensive enough that the
30-minute firings could never repeat it inside one night. Fail-fast made it cheap, and
cheap plus unbounded is a loop: over one simulated Saturday night's ~16 firings a
degrading summary ran TEN times, writing ten `degraded_no_narrative` rows and ten export
sets for a single session. A forced daytime run records `skipped`, which is not in
`ledger.ATTEMPT_STATUSES`, so an operator trying at lunchtime does not burn the night's
attempts.

## TJ-13B - a model nobody has ever run has no reserve (2026-09-19/20)

The long form behind the CLAUDE.md clause *"A model nobody measured has no reserve"*.
plan.md §12.4 TJ-13 item 7, decision 0021 answer 20. Branch `claude/tj13b-large-local`
(tip `800ecb8c`), **merged into `lead/p033-integration2` as `e54c8203`** - not on `main`
yet, so nothing below is live on the desk.

**What was measured.** In the 476 live ledger rows read on 2026-09-19 the only models
named were `gemma3:12b`, `gemma3:12b-tbv3ctx` and `gemma3:12b-tbv3ctx-64k`.
`ai_local_model_large` - a Gemma-3 27B Q3_K_M GGUF tag - was configured and had **never
run**. The week story is supposed to be written by it, and `plan.md` §12.3 says a slot
declares a `reserve_minutes`; so the one number that decides whether the week story may
start at all could not be written down by anybody. Guessing it is the failure mode the
reserve exists to prevent: a slot with an invented reserve either skips a window it would
have fitted in, or runs a 27B into the opening bell. (The `RETIRED (2026-08-10)` line the
CHANGELOG carried for this tier was a judgement, never a measurement; TJ-13B replaces it
with one - and the 27B has STILL never run, so nothing here claims it fits.)

So the number is MEASURED, once, by the trader, and read back from the ledger.

* **The probe is a model LOAD, and every rule about model loads applies to it.**
  `scripts/ai_jobs/model_probe.run_model_probe`, reached as
  `run_ai_jobs.py --probe-model large`, is a COMMAND and never a slot: it builds no
  slate, runs no other job, and exits 0 when it measured, 1 when it refused with a
  printed reason. There are exactly THREE refusals - it is not night; an AI job is
  running; this session was already measured. `PROBE_RESERVE_MINUTES` is 45, so it
  refuses from about 05:15 PDT.
* **The lock is the DEFAULT guard, not a caller's option.** `lock` defaults to the
  sentinel `USE_RUNNER_LOCK`, and `None` resolves to the same `runner_lock` - there is
  deliberately **no value of `lock` that means "unguarded"**, because for one day the
  guard was a caller-supplied parameter, which made the most dangerous call in the
  package the one that named nothing. The guard is HELD across the whole measurement, so
  a 22:30 firing under it stands down cleanly. Where `ai_jobs.runner` runs unguarded on a
  box with no exclusion primitive (what that protects is a night of cheap deterministic
  work), the probe REFUSES instead: uncertainty is not confirmation when the cost of
  being wrong is a second model load beside a working 12B on a 32 GB box.
* **`--force` re-spends ONE check.** Only "already measured for this session". It does
  not buy the clock and does not beat a held lock. Round 1 of the review was NO-GO
  because the flag never reached the probe at all, so the probe's own refusal - *"pass
  --force to measure it again"* - was advice the command could not take.
* **It reads a COPY.** The newest week of fact packs is copied to a temp folder, read
  from there, and deleted. The live store is read once, to list and copy, never written.
* **The row is `manual_test`.** ONE ledger row per measurement, `job="model_probe"`,
  carrying load seconds, tokens per second, `peak_memory_mb` (the MACHINE's in-use
  memory, with `baseline_memory_mb` beside it, never a delta) and
  `context_tokens_accepted` (the SERVER's own `prompt_tokens`, beside what the desk
  CONFIGURED - the gap between the two is the finding). `manual_test` never counts as
  session coverage, and the row's `session_date` is the LAST session, so a Saturday-night
  probe is filed under Friday. That is right, not a bug.
* **The numbers say what they are.** One call cannot split the weight load from the
  prompt evaluation and the generation, so every row names its `basis`:
  `single_call_end_to_end` (load an upper bound, throughput a lower bound - a reserve
  derived from them is conservative by construction) or `server_reported_timings`.
* **No measurement means no large model, never a default reserve.**
  `reserve_minutes_from_probe` is `(load + 3,500 tokens / measured rate) x 1.25` - 600 s
  at 10 tok/s gives 19.8 min - and returns `None` when the tier was never measured; a
  MEDIUM probe never answers for LARGE, and the newest row wins.
  `provider.week_review_plan` then answers `may_run_large: False` with its reason and
  names the MEDIUM model: the trader wants a week story every Saturday, so a missing
  measurement costs the large model, not the story. It writes no row.
* **A fallback that cannot be read is a silent downgrade.** The ledger's `model` column
  is one string, so a row reading `gemma3:12b-tbv3ctx-64k` cannot otherwise be told apart
  from a night the large model was never attempted. `provider.request_with_fallback`
  sends the IDENTICAL closed schema to both tiers, rejects an invented `source_id` whole,
  publishes NOTHING when both fail, and returns asked / answered / why for the SLOT to
  write under `model_attribution`. It writes no ledger row itself (`ledger_path` is
  accepted and unused by design), and it RAISES `ValueError` for `openai` before anything
  is sent - TJ-5's week slot must catch that and record a FAILED row. `local_large` is a
  name in this seam only, never in `ai_summary.normalize_provider` nor `ai_credentials`,
  whose vault knows openai and anthropic and raises on anything else.

**Two lessons from the integration, both about tests rather than models.** A guard pinned
"byte-for-byte to the branch's base" (`test_tj13b_local_large_provider.py`'s weeknight
slate test) went red the moment the lead's `1f260ffa` fixed slot order on `main`; it was
re-pinned to main's order on the lead's authority - same 20 names, positions 12-14 - and
**TJ-10's coming `read_grades_mature` slot must be added to that tuple at integration.**
And while the `ai_jobs_runner` lock is held, about 42 tests that call the real
`runner.run_slots` FAIL rather than skip, so a suite run inside the AI window is neither a
baseline nor proof that a branch is red (`docs/AGENT_TEAM.md`, "The nightly AI lock").

## The four overnight repairs TJ-13A carried - envelope, membership, examples, truncation (2026-09-19)

**The envelope this code asked for** (TJ-13A item 5). `journal_enrichment` failed
2026-09-15/16/17 with "tradingbot_trade_enrichment is missing required field(s):
confidence, sources, summary, tags, unknowns" - ALL FIVE, which only an object carrying
none of the contract's keys at the top level can give. The request is where that shape
comes from: the payload sets `response_format.json_schema.name`, and a backend that
echoes its own envelope key returns the complete answer one level down.
`unwrap_schema_envelope` unwraps it only when the object has EXACTLY ONE key, that key is
the schema name the request supplied, and its value is an object. A reply that echoes the
SCHEMA has several top-level keys, carries no answer, and stays rejected - an
answer-shaped hole is a failure, never a row.

**A membership-only name is counted, not printed** (TJ-13A item 4). Measured on the live
`ai_morning_brief.txt` 2026-09-19: `Analyzed 53 of 312. Membership-only 259.` - 259 of
312 sections said only that the symbol is on a list, the one thing the reader of a
watchlist already knows, and they pushed real briefs past the 48 KB ceiling into "omitted
from this small summary file". The name stays in the header's `Membership-only` count,
which is why that header carries three numbers rather than one. `_morning_section` keeps
Q3.3's prefix rule for the one entry it renders; the morning file no longer reaches it.

**An example list names three different things** (TJ-13A item 6). The 2026-09-17 measured
report read `ABCL` three times at 6.1619 percent, `ERAS` three times at -7.5875 and one
swing occurrence id three times at 4.4529 R, out of 19,045 and 309 measured observations:
a symbol carries several observations in a session and `_example_tables` took the top
three ROWS. It now de-duplicates by name (occurrence id for a swing row) first. The
DENOMINATOR does not move - the label is about the full population - and a short list
stays short rather than re-admitting a duplicate to reach three.

**A truncation flag describes THIS package** (TJ-13A fix round). `build_ticker_evidence`
builds a per-symbol package by copying each session source (`dict(raw)`) and replacing
its content with the symbol's slice - so the session's own `truncated` flag rode along
onto a projection that was never itself cut. Measured read-only over all 53 published
packages of the 2026-09-18 night: **80 of the 133 per-ticker sources carried the flag**,
packages ran 6,476-9,675 chars against a 22,000 per-item budget, and every flagged source
was AT MOST 825 characters against a 16,000 `MAX_TICKER_SOURCE_CHARS`. Neither budget
constant was the cut - all 80 were false. The live `ai_morning_brief.txt` says
"truncated" dozens of times (`grep -c` gives 63 lines, `grep -o` 64 occurrences; the
point is the order of magnitude, not the digit) and the model hedges accordingly -
"truncated, limiting the scope of the analysis" - about a 107-character source it had
received whole.

`_projection_was_cut` re-derives the flag from the projection in hand: True when the
mapping branch produced a `truncated_record`, or when the encoded projection reaches
`MAX_TICKER_SOURCE_CHARS`. A projection that lands exactly ON the ceiling is reported as
cut, which is the conservative direction - "you may be reading part of this" is safe to
say wrongly and "you have all of it" is not. A symbol that genuinely has more rows than
the ceiling holds still says True.

## TJ-15 - what the misses had in common, and the two floors it took (2026-09-19, packet TJ-15)

The long form behind the CLAUDE.md rule *"A miss contrast has TWO floors and judges D1
only."* plan.md 12.4 TJ-15, decision 0021. Branch `claude/tj15-miss-contrast` (tip
`8547c4a5`), merged `fb3f55e9` into `lead/p033-integration2`, with the lead's slot-position
fix `1f260ffa` on top. Two reviews by reproduction on the real 2026-09-18 window: NO-GO,
then GO after the fix round.

TJ-11 put the names the trader turned down that ran onto the Day Review page. TJ-15 asks
the next question of the same rows - *what did they have in common?* - and answers it with
arithmetic only. **No model is called anywhere in this chain.**

**One method, built once.** `scripts/evidence_contrast.py` takes two groups of
point-in-time feature mappings and returns, per feature, the two integer counts, the two
medians and ONE rank statistic: `compression_calibration.auc`, the statistic PCT-3
established, CALLED rather than re-derived. The rank key is `abs(auc - 0.5)` descending
then feature NAME ascending; **no group size, no median magnitude and no R statistic may
enter it** (gate #43). An AUC below 0.5 therefore ranks on SEPARATION, not on being high -
live, the `like` group's leader scored 0.33. A feature only one side measured is NAMED in
`unmeasured_features`; a blank cell is left out of the median and never read as zero.
`rate()` is the ONE Wilson - `swing_headline.WILSON_Z` through `walkaway_day._wilson` -
over CLOSED horizons only, with the open ones counted, printed as `pending`, and in neither
half. TJ-16 reuses this module; it is not TJ-15's private helper.

**Two floors, and they are different floors** (fix round). The first build had one.
`MIN_REPORTABLE_N` gates a group's RATE, whose denominator is every measured decision - but
a FEATURE is measured only on the decisions that also carried a point-in-time scan row, and
on the live window that was a much smaller number. `veto / sma_incoming` had n=47 and
measured=30, cleared the rate floor, and then named the night's leading finding off **four
rows against one** at an AUC of exactly 1.0 - which is what four-against-one gives whenever
the four sit above the one. So `MIN_CONTRAST_SIDE_N` is 10: a feature is ranked only with
at least ten rows on EACH side and `MIN_REPORTABLE_N` across both. One under that is still
NAMED, with both counts, in `thin_features` - hiding it would say it was never looked at -
carries no AUC, and never enters `features`. `compared` counts the RANKED features only and
the statement prints both numbers. A GROUP is a leader only with a reportable rate AND at
least one ranked feature; one with a reportable rate and no ranked feature keeps its row
and its rate and reads `no feature had enough rows on both sides`, and the pack's headline
says that rather than claiming the floor. `n_a` and `n_b` are FIELDS at the group level and
on every feature and thin-feature row, never only in prose.

**D1 only, because the ruler and the features are D1's** (fix round). The first build
pooled every timeframe, and the population was not what it was assumed to be: over the 20
sessions ending 2026-09-18 it was **1,026 M5 against 1,013 D1** (plus 4 H1 and 2 "5M"), and
the top-named leader was an M5 group judged on a five-session swing horizon with D1-scan
features it never had. The pack now judges D1 and nothing else, compares the timeframe
case-insensitively, COUNTS the rest in `excluded_by_timeframe` and says the number in its
own sentence; a row with no timeframe is counted in `no_timeframe` and never read as D1.

**Point in time, and one session back is still point in time** (fix round). The features
are the LAST scan row for `(symbol, side)` at or before the decision's own stamp - never a
later one, never an older one. Restricting that to the decision's OWN session threw away
47-83% of decisions, because the scan does not run when the trader clicks: on the live
window it ran at roughly 07:00-07:50, 10:01, 12:45 and 13:00 desk time and on some sessions
only once, so an evening or pre-market decision had no same-session row at all. The window
reaches back ONE exchange session (`MAX_SCAN_AGE_SESSIONS = 1`) - what the trader was
actually looking at in the evening or before the open - with the age carried on the row and
counted per group as `scan_same_session` / `scan_prior_session` / `no_point_in_time_scan`;
live that was 238 / 140 / 97 of 475 measured decisions. `run_timestamp` is NAIVE desk wall
time and an annotation's stamp is ZONED, so the desk zone is ATTACHED to the naive side
through `ui.annotations.pass_bars.attach_desk_zone` and the aware side is never stripped -
strip it instead and a `14:05+00:00` spelling of a 07:05 Pacific veto reads 14:05, which
puts the 07:30 scan *before* the decision.

**Only a VETO has a reason code** (fix round). A veto's reason is a code from the versioned
veto vocabulary, so vetoes group by it, pooled across vocabulary versions and said so in
the pack - a code is never reused, so a code means one thing. Every other verdict's reason
is the trader's FREE TEXT, and grouping on it made one group per sentence: live, `dislike`
split into two groups of one, each immune to any floor because a group of one is never
compared with anything. They group by the verdict alone.

**Where the slot sits, and why that is not a free choice.** `miss_contrast` is registered
INSIDE the deterministic stage, after the cohort graders and `theta_pick_grading` and
**before the `market_story_rollups` + `measured_report` pair that closes it**. The packet
first appended it after that pair; the lead moved it above them at integration
(`1f260ffa`), because WS-10D and WS-RP each pin `measured_report` DIRECTLY after
`market_story_rollups` and `test_veto_cohort_grading` pins the whole slate - three order
assertions the packet's targeted runs never reached and the full suite caught at once.
The position is load-bearing in the other direction too: `runner._STAGE_ONE_LAST_SLOT` is
`measured_report` and `_deterministic_stage` walks the slate up to and INCLUDING that name,
so a slot appended AFTER it is not in stage 1 by that function's reckoning and silently
leaves the **Sunday** slate, however deterministic it is. `_STAGE_ONE_LAST_SLOT` is
untouched, `measured_report` still closes the stage, and the slot runs on the weeknight,
Saturday and Sunday slates alike. Nothing here reads the measured report and nothing there
reads this pack, so only the ORDER was ever a choice - and only one of the two orders runs
on a Sunday. `uses_model=False`, `max_attempts=3`, `reserve_minutes=5.0`. Decision 0018
carries the slot-list addendum.

**The file, and what a real pass costs.** `d1_features_history.csv` is ~709 MB over 264
columns, streamed by session through `stream_feature_rows` as a GENERATOR: ONE
`csv.DictReader` pass with a 20-session filter is **7.8 s warm / 17.7 s cold** at about 100
KB of traced Python allocation, where `pandas.read_csv` of the whole file would be roughly
1.4 GB inside the process that owns the night. The slot holds one feature mapping per
DECISION, never a list of the file. End to end on the real 2026-09-18 window: 600-900
durable daily frames read, a ~119 KB pack, **~18-24 s and ~240-300 MB of process heap** for
the whole slot. A live group compares 54-81 features and names another 6-88 too thin to
call. The re-derived live result: 1,013 D1 decisions judged in 11 groups, leaders
`compressed` (`last_volume`, 32 v 71, AUC 0.72), `veto` with no code recorded
(`rs_vs_industry_5d`, 15 v 53, 0.70) and `like` (25 v 59, 0.33).

**One rule, one mapping.** `real_miss.verdict` is CALLED through its module attribute, and
which session a decision belongs to is `walkaway_day._row_session` - the same seam Day
Review's own walk-away asks - so the table under the misses and the misses themselves
cannot disagree, and TJ-11F's judged-session rule carries through unchanged.

**It never costs the night.** A missing, locked or unreadable features file, an unreadable
annotation store and an absent daily-bar store are each a recorded REASON on an `ok` row.
The one thing it REFUSES is a missing, empty or unparseable `session_date`: that is
`failed` with a reason and NO file, because a pack keyed to nothing publishes
`miss_contrast-.json`, one file every later session would supersede and no reader could
date. The pack goes to `store.digests_dir()`, the same place `daily_digest` and
`measured_report` publish - **the DAS on this desk**, not a local-first write - gated by the
runner's `store_available` probe before any slot starts, and written temp-and-rename onto a
superseding sibling, so a correction is a new file and an earlier pack is never rewritten.

**Advisories, recorded and not repaired.** The no-code veto bucket reads just `veto` where
it wants to read `veto (no code recorded)`. The pack never states the DIRECTION of a
leading feature in words - it gives the two medians and leaves the reader to see which way
they point. `MIN_CONTRAST_SIDE_N = 10` is lenient: a merged-tree like leader rested on 10
against 37, so gate #160 watches whether ten a side is enough. `read_latest` is a FILE
READ - TJ-5's Week Review table must call it on a worker, never on the Qt thread. And the
builder calls six PRIVATE `walkaway_day` seams deliberately, rather than copy a rule that
would then drift.

**One lesson for the next packet.** The packet's targeted runs were green while three order
assertions elsewhere were red. A packet that adds or MOVES a runner slot runs `-k "slot or
stage or slate or order"` plus the SEVEN order pins - `tests/test_ai_jobs_runner.py`
(`EXPECTED_SLOT_ORDER`), `tests/test_veto_cohort_grading.py`,
`tests/test_ws_10d_market_story.py`, `tests/test_ws_rp_shared_report.py`,
`tests/test_tj13b_local_large_provider.py`, `tests/test_tj13b_probe_guards.py` and
`tests/test_opt_in_evidence_scopes.py` (the seventh, amended by the lead 2026-09-20 at
TJ-16's merge: the `ai_summary` → `ticker_briefs` pair keeps its ORDER and only
`day_review_narration` and `observation_tags` may sit between them). plan.md 12.3 carries it.

## TJ-16 - a skill number, and words tagged blind (2026-09-20, packet TJ-16)

The long form behind the CLAUDE.md rule *"The ledger reads beside three baselines on the
same stamps, and the tagger never sees an outcome."* plan.md 12.4 TJ-16, decision 0021.
Branch `claude/tj16-prediction-contrast` (tip `d0d61f95`), merged `c4a760e5` into
`lead/p033-integration2`. Two reviews by reproduction: NO-GO, then GO after the fix round.

Two questions, one packet. *What leads to a good call?* and *what were you actually looking
at when you said it?* The math finds the tendency; the model labels the words; neither may
do the other's job.

**Why a baseline and not a hit rate.** A trader who is right 55% of the time reads like a
coin flip until you know what the coin was. On the fixture the tester built - ten sessions
of four hourly clicks - the trader went 22 of 40 (55%), and the three naive rules answering
the SAME forty stamps went 24/40, 32/40 and 32/40. All three beat them. That sentence is
only sayable because the baselines are measured on identical questions:
`market_read_grades.baseline_reads` re-answers each row from that row's own point-in-time
context snapshot and no bars at all, so a naive rule cannot see anything the trader could
not. The trader's verdicts are STORED and are never re-measured; a baseline has no stored
verdict, so its verdict is measured now - through `market_read_grades._verdict_for`, the
module attribute, never a copied `abs(move) <= 0.25`. Swap the band and every baseline
follows it while the stored rows stay where they were, which is exactly what a superseding
band needs.

**A baseline that cannot answer is not a baseline that lost.** `compressed` is not a
direction, so `with_the_d1_environment` has no answer on those stamps and they LEAVE its
fraction. Counting them as four losses for the naive rule would have flattered the trader
with rows nobody measured (plan.md sec 5).

**Calibration says the unflattering thing.** `high_beats_low` is a bool, and when it is
False the statement reads "High did not beat Low" with both rates and both `n`. It is
`None` - "not answerable yet" - while either bucket is under `MIN_REPORTABLE_N`, which is
not the same as "no".

**The contrast is TJ-15's, called.** `evidence_contrast.contrast` already holds the rank key
(`abs(auc-0.5)` then feature NAME), the two floors and the sentence. TJ-16 encodes and
splits; it computes no AUC of its own. A `flat` reading is counted on the horizon and is in
NEITHER group: the market did nothing, so the trader was neither right nor wrong, and
folding those rows into either side would be inventing a verdict.

**The encoding is where an honest pack is won or lost.** A numeric field keeps its own name
(`hour`, `gap_pct`). A categorical one becomes one feature per value -
`spy_vs_prior_range:above` - worth 1.0 on a row holding that value, 0.0 on a row holding a
DIFFERENT MEASURED value, and **nothing at all** on a row whose field reads `unmeasured`. On
the fixture eight of forty reads had no `spy_vs_prior_range` reading: the feature is ranked
on 32 rows (18 right against 14 wrong), and a builder that had read the eight blanks as
zeros would have ranked it on 40 and moved the statistic with rows nobody measured. A field
that parses as a number ANYWHERE in the population is numeric everywhere, which is why
`gap_pct` - measured on five of forty - keeps its name and lands in `thin_features` with its
two counts and no AUC rather than becoming a category with thirty-five "unmeasured" members.

**Nothing is hidden and nothing is ranked by result.** The contrast is called with a `top`
that covers every feature over the FEATURE floor, because the context block is about a dozen
fields and hiding nine of them would be a bounded view with no reader. The pack's bounded
view is `tendencies`: at most three cells, each citing its own table, key and `n`, ordered by
`n` descending then by name - a SIZE rule (gate #43) - and a cell under `MIN_REPORTABLE_N` is
never offered however quotable it reads. Two weeks of clicks is forty rows and every cell of
it is under the floor, so the week story may narrate NOTHING from it; the most quotable
sentence in that pack ("you read the 07:00 hour better", 7-3) is the least supported one.

**The tagger is blind by construction.** `build_evidence` knows about two strings and a
picklist and has no way to reach anything else, so there is no verdict key to remember to
delete at the end - the grade ledger of the very same session sits on disk beside the job and
none of it, not a grade id and not a price off the tape they were measured on, reaches the
prompt. A tag derived from the outcome is not a label of the words, it is a rationalisation
of the result. The grounding rule is `market_thesis`': a span is a QUOTATION and
`text[start:end]` must equal `quote` exactly. One row that does not reproduce rejects the
WHOLE reply - not the row - because a half-accepted answer is a file nobody can trust and
nobody can tell apart from a whole one.

**A JSON schema is a grammar hint, never a guard** (review round 1). `MAX_TAGS` (60) and
`additionalProperties: false` lived in `TAGS_JSON_SCHEMA`, which is sent to the provider in
`response_format` - and `ai_summary._request_local_summary` already ships a documented
fallback for a backend that refuses to compile that grammar, while
`validate_structured_output` walks only the TOP level and only declared types, so an array of
objects arrives untouched. The reviewer handed the slot a 10,000-row reply and watched it
publish `ok`, and a tag row carrying `"why": "smuggled"` believed. `verify_reply` now
re-checks the SHAPE as well as the grounding, and each rule throws the whole reply away: more
than `MAX_TAGS` rows, a key outside `REPLY_KEYS` at the top, a key outside `TAG_KEYS` on a
row, or a byte-identical duplicate row (identity `(note_id, code, start, end)`, so key order
cannot dodge it). **A model that repeats itself has not been verified**: de-duping silently
would have stored a file that does not say what the model returned, and `codes_by_entry`
collapses codes per entry, so no number would have moved and nobody would ever have noticed.
Two rows on ONE note with OVERLAPPING spans and DIFFERENT codes stay accepted - one sentence
can cite a level and be hedged, and that is the finding, not an error. **The lesson is not
TJ-16's alone**: TJ-4's first build shipped the same trust in the same week.

**The window is asked of the ledger, not applied after it.** `EvidenceLedger.read` filters by
`session_date` while streaming; the first build read every row ever written and filtered in
Python, which is not what "never stream the whole journal unbounded" means. It is safe to
narrow because a CORRECTION carries the ORIGINAL `session_date` ("`session_date` is what the
entry is ABOUT"), so both halves of a supersede pair stay inside a one-session window and
`resolve_entries` still hides the older one. Measured by the reviewer on a COPY of the live
stream (`market_journal-202608.jsonl` + `-202609.jsonl`, 174,075 B, 17 sessions): 84 rows
unwindowed against 7 for one session, 0.0059 s and 0.17 MB peak, identical answer - and a
supersede pair inside the window still resolves to the CORRECTED text while another session's
note never reaches the payload.

**Hindsight is labelled, not dropped.** `prediction.because` can carry an outcome - the
reviewer wrote "in hindsight the 50 day failed and I lost on this call" and watched it reach
the payload verbatim. The machine adds no outcome and the trader's own words are the artifact
under study, so the note is tagged like any other and nothing it produces is re-ranked (the
ten-session fixture built with and without the label is byte-identical across every horizon
number). Each stored tag carries the entry's computed `written_after_the_session`, the tags
file header counts `entries_written_after`, and `prediction_contrast`'s `tags` block carries
the same count, present and zero, never absent. The label stays OUT of the payload: a prompt
that said "this was written after the close" would be telling the tagger something about the
outcome, which is the one thing this slot may never do.

**Its vocabulary has its own loader, deliberately.** `ui/annotations/vocabulary.py` is a
generic family loader whose `_parse` demands a unique single-character `hotkey` and a boolean
`note_required` per entry, and it is called by the capture rail on every click. A tag
vocabulary has neither field, and widening a capture-rail loader for a nightly reader is risk
for no gain (lead decision, 2026-09-20). `observation_tags.load_vocabulary()` reads the
highest `observation_tags_v*.json` and refuses a file whose declared `vocab_version`
disagrees with its own FILENAME, so a v2 ships beside v1 and rows stamped v1 stay
interpretable against exactly the list that produced them. No test asserts the version.

**Tonight's tags are tonight's.** The tagger is a stage 2 slot and the contrast is a stage 1
one, so a night's codes reach a contrast on the NEXT run. The pack proves it by holding no
`tag:` feature at all on the night it was tagged - and `tags.reads_matched == 0` beside a
non-empty `codes` list is the live signal that the `entry_id` join is broken, which is what
gate #166 reads.

**The empty state is the state the desk is in.** Measured 2026-09-20:
`C:\TradingBotData\day_review\` holds only `sessions\` - there is no `reads\` folder, so
there are **ZERO clicked grade rows and zero notes carrying an observation**. The first thing
the trader sees is therefore the honest empty state: every surface opens on `no clicked calls
yet`, nothing raises, no rate is 0.0 ("you are never right" is not what "nothing measured"
means), and the slot writes an empty pack and records `ok` (or `manual_test` on a forced
run), never `failed` - an evidence job is never allowed to cost the thing it records.

**Advisories, recorded and not repaired.** `prediction_ledger.your_reads` names the baseline
with the most RIGHT rather than the best RATE - the line prints both integers so it is not
misleading, but "best" by count is not the comparison the docstring implies; it is batched
for TJ-12, which owns that page. And a caller that passes `tags=` explicitly while leaving
`written_after` unset reads `entries_written_after: 0` - "present and zero" for a fact the
build did not measure; only the test-injection path does this, the nightly default deriving
both from one read of each night's tags file.

## M1 - a shadow that measured nothing for ten days (2026-09-05)

The long form behind the CLAUDE.md rule *"The AVWAP band challenger is measured through
the CATCH-UP path too, and its view names its own coverage."* Governing spec:
`docs/AVWAP_BAND_VARIANT_STUDY.md` T3/T4, plan.md Phase 0.10 and 0.19.

### What was measured, on `main` @ `e7b12ebe`

- `master_avwap_band_variant_stats.csv`: **40 rows, 11,292 setups, `n_variant = 0` on
  every row**, `n_variant_unmeasured = n` on every row, all four `_variant` columns 100%
  blank. Written every scan since 2026-08-26; never once a comparison.
- Every tracker record: `"reason": "no band-variant block on the scan entry"` - **186 of
  186 setups** on scan_date 2026-09-03.
- `master_avwap_ai_state.json`, same scan: a full block
  (`avwap_bands_oneoption_bb20_v1`) for **all 423 symbols**. AAON carried stdev 4.72 on
  anchor 2026-08-10 in the AI state and "no block" on its setup record for the same
  anchor date. The scan was computing the challenger and throwing it away.
- The catch-up path ran with real work twice on 2026-09-04 (38.4 s and 14.1 s in
  `trading_bot.log`).

### Why

Two builders for one symbol entry. `runner.py`'s live scan sets
`current_anchor_variant` / `previous_anchor_variant` beside `current_anchor_meta`, from
the same frame and index. The **persisted tracker on a normal day is not written by that
path**: it is written by the staleness catch-up
(`backfill_setup_tracker_from_recent_sessions` → `_evaluate_priority_snapshot_for_date`,
`docs/DURABILITY_CATCHUP_PLAN.md` §2.1), which builds its own ~100-key entry inside
`legacy.py`. `build_anchor_band_variant_meta` lived in `runner.py`, which imports
`legacy` - so the catch-up could not have called it even if someone had thought to.
`build_tracker_setup_record` then did exactly what it was written to do: stamped the
placeholder block that says a caller predating the shadow handed it nothing.
`_find_tracker_stop_candidates` saw no variant anchor and added no `band_variant` stop;
`_build_tracker_scenarios` built no variant scenario; `_band_variant_paired_scenarios`
found no challenger to pair; `build_band_variant_stats_rows` counted the setup under
`n_variant_unmeasured` and moved on. Every link behaved correctly and the chain measured
nothing.

### The rules this produced

1. **One function, two call paths - never two copies of a formula.**
   `build_anchor_band_variant_meta` now lives in `legacy.py` and `runner.py` re-exports
   the name, so the live scan is byte-identical and the catch-up computes the identical
   block. A third builder for a symbol entry would reopen this defect; do not add one.
2. **A placeholder that becomes the normal case is a defect, not a default.** "No
   band-variant block on the scan entry" is for a replay or an old payload. It is now
   **unreachable** from the catch-up path and a parametrized test asserts it over both a
   long and a short frame. A frame too short for the 20-close window says
   `"fewer than the lookback's closes before this bar"` - a stated reason, never a zero
   band and never the placeholder.
3. **A surface that shows an empty comparison must say it is empty.** The Band variant
   tab renders `Measured N of M setups (K unmeasured: <top reason>).` above the table,
   as a pure function of the export's own `n` / `n_variant` / `n_variant_unmeasured`
   sums. The table alone could not carry this: a family, a side, a champion R and four
   blank cells read as *"no difference"*, which is the opposite of *"never computed"*.
4. **The reason is aggregated where the records are, not where the panel is.**
   `master_avwap_band_variant_stats.csv` gained exactly one column,
   `top_unmeasured_reason`, because the reasons live on the tracker records and the live
   tracker JSON is **1.1 GB** - a panel that opened it to name a reason would freeze the
   Qt thread. Reasons are carried verbatim; a machine never re-codes a stated reason, and
   "the block was never handed over" and "the window was short" are different defects
   with different fixes. Ties break alphabetically so the cell is stable run to run.
5. **A tracker record is rebuilt on every persisted write.** Existing records pick the
   block up on the next write, so nothing here ever needs a migration.
6. **The champion did not move, and the fixture is what proves it.** The B-2 parity
   fixture was frozen before the shadow existed and is never regenerated from the code it
   pins - a fixture generated by the code it pins is a self-portrait.
7. **An evidence clock starts when evidence starts.** T4's ≥ 20 sessions of forward
   accrual and ≥ 40 finalized setups start at the **first measured row**, not at the date
   the shadow shipped. Ten days of zero rows are not ten days of accrual.

### Storage, re-measured

5,590 bytes per record (+16.5%: 33,813 → 39,403) on the M1 fixture: two anchor blocks
plus four challenger scenarios on the four baseline exit templates. B-2 measured 9,982
bytes on its own fixture before the experimental templates were excluded. Either way the
study's "a few hundred bytes per setup" is an order of magnitude low. At ~186 setups per
scan date the cost is roughly 1 MB per session, on records written from now on.

---

## M3 - the tracker that could not be written, and the setup nobody measured (2026-09-05)

Three findings from the lead's measurement audit of 2026-09-05 (~02:00 PT), all
authorized by the trader with *"Fix all of these failures"*.

### The purity gate was refusing the trader's own decision

`local_settings.json` carries `daily_bars_source: "yahoo"` — the R10.0b §1.3
interim pin. The durable daily-bar store is mixed because IB returns
regular-session volume in ROUND LOTS (`useRTH=1`, `whatToShow="TRADES"`) and
Yahoo returns the full consolidated session in SHARES, and the observed ratio is
symbol-dependent (SPY 1.0x, TSLA 56x, AAPL 81x, A 162x, NVDA 188x), so no
constant converts one into the other. That is why it is a pin and not a rescale.

The setup tracker's purity gate was written in July 2026 against a completely
different problem: an IB client collision routing every symbol to Yahoo behind
the scanner's back. Its shape is a small dirty tail QUARANTINED (1-2 chronic
symbols like LC and BF.B always fall back) and a large non-IB fraction VETOING
the whole write, because that is the systemic-fallback signature. It had no way
to tell a fallback from a declaration.

So with the pin in force it refused the scheduled write every day. The 2026-09-04
13:00 run logged "WITH setup-tracker write" and then, at 13:03:59, *"Setup
tracker refresh skipped for this mini-PC run because tracked setups used
non-IBKR daily data"* over 139 symbols with `sources=cache`. The tracker JSON and
its SQLite mirror had last been written at 07:46-07:47 that morning — by the
staleness catch-up, a synthetic replay from stored daily bars. **A recovery path
had quietly become the routine writer.**

The fix is narrow. The pinned source is a source of record; a symbol on it is
PURE. `cache` and `unknown` are accepted ONLY under a pin and only as the absence
of contrary evidence — `daily_bar_provenance_for_source` deliberately refuses to
map a cache read to a volume unit, because reading a row off disk tells you it
came off disk and not what wrote it. Where the frame carries per-row provenance
in the `source` column, that is what is read, so a store holding pre-pin IBKR
rows and post-pin Yahoo rows is judged on what it actually says. Anything that is
neither IBKR nor the pin is still a fallback nobody declared and still vetoes at
the same 20% fraction. **With no pin the gate is byte-for-byte the July one**,
and two of the tests written for this packet were green before the fix precisely
to hold that.

### Two clocks, and the page was showing the wrong one

The Setup Tracker page read one mtime across eleven exports. The
`scan_factor_*` files are rewritten by every scan; the tracker's own snapshot is
rewritten only by a pass that actually replayed it. On the days the write was
refused outright, the page therefore claimed to be as fresh as the last scan.

`saved_at` (market-local) and `saved_by` (`close_slot` / `catch_up_backfill` /
`manual`) now ride on the payload, and the three stats CSVs carry
`tracker_saved_at` / `tracker_saved_by` so the panel can name the snapshot's
clock **without opening the 1.1 GB JSON**. The status line is
`Tracker as of <saved_at> (<saved_by>); scan factors as of <mtime>`.

Two details that are load-bearing rather than incidental:

* the stamp is passed to `export_setup_tracker_views` explicitly, because the
  export runs BEFORE the save — reading `saved_at` off the payload there would
  stamp the CSVs with the previous save's clock;
* `load_setup_tracker_payload` names both keys. It rebuilds the payload field by
  field from a fixed default, and that is exactly how `data_session` was once
  written and dropped straight back out, leaving the whole vintage fix inert in
  production.

`tracker_store.HEADER_FIELDS` gained the two keys as well: the mirror FOLLOWS the
JSON (decision 0017), so a header key the JSON carries and the mirror does not
would be a parity difference `verify` reported forever — which is gate #57's
entire measurement.

### A setup that stops being measured is not a loss

37 OPEN setups were older than 20 sessions on 2026-09-04, going back to
2026-05-06 (CTRA, KALV, MU, PWR …), several with scenarios whose `last_action`
still read "Awaiting update" — never replayed since creation. 41 more had
`open_scenario_count == closed_scenario_count == 0`: some with an empty
`scenarios` dict (CLF, OKLO, GNTX), some whose every scenario was experimental or
band-variant (MU 06-04, GBTG 07-01 carry 12-18 scenarios and not one baseline).
`setup_status` was pure scenario closure with no time term, so all 78 sat in
denominators as though they were evidence.

`EXPIRED_UNMEASURED` is the third answer. `expiry_reason` is
`no_replay_20_sessions` (more than `TRACKER_STALE_SESSIONS` = 20 exchange
sessions since `last_replayed_session`, counted with
`market_calendar.trading_days_between` — weekday arithmetic counts Thanksgiving
as a session) or `no_baseline_scenarios`.

The rules around it matter more than the status:

* **It is applied AFTER the closure rule**, so a setup that closes normally is
  never expired.
* **Uncertainty never deletes.** A date the calendar refuses, an unparseable one
  or a missing one all leave the record exactly where the closure rule put it.
* **It runs in the recompute AND as a sweep.** A setup whose daily frame comes
  back empty — a delisted symbol, a fetch that failed — is skipped before
  `recompute_tracker_setup_record` is ever called, and those are precisely the
  "Awaiting update" records the rule exists for.
* **It leaves numerator and denominator both**, and every export carries
  `n_expired_unmeasured` beside its `n`. An exclusion nobody can see would be a
  second version of the defect it fixes.
* **Nothing is deleted and only the closure rule un-expires a record.** A
  replayed scenario comes back through the recompute as OPEN or CLOSED and the
  reason is cleared.

### What the reviewer caught, and why each one hid

Three blockers on the first build, all of them the same species: a number that
looked right in a unit test and was wrong on the live file.

**Two clocks, two zones.** The point of the two-clock line was that the tracker
snapshot and the scan factors have different ages. `saved_at` was rendered
market-local with an offset and the scan-factor mtime machine-local with none,
so on this PT desk the pair read *three hours apart for the same instant* — the
line invented a staleness it existed to disprove. Both are market-local ISO with
the offset printed now. The test stamps both from ONE instant and asserts the two
rendered strings are equal, because asserting a format would have passed.

**An all-expired group took its own count with it.** `if not rows_for_group:
continue` looked like a guard against an empty row; it was actually the branch
that dropped `expired_rows`. `build_tracker_stats_rows` had the identical dead
shape one level down (`grouped.setdefault(key, [])` followed by `if not rows:
continue`). Every unit test passed because every fixture had at least one
surviving record per group. On the live 2026-09-04 mirror the sentence said
**16 where 45 records had expired** — under-reporting by exactly the groups that
were worst, which is the direction that hides a problem. Both builders now emit
the row: zero measured setups, blank measures, the real count. The stats builder
needed a `representative_by_group` to do it, because `rows[0]` does not exist
when every row in the group was expired.

**A gate nobody could satisfy.** The first #69 asked for `n_expired_unmeasured
>= 37 + 41` from the audit's prose. Those two numbers counted different things
from what the implemented rules count, so the gate could not pass however
correct the code was. It is restated to numbers reproducible from the mirror
(setups 32 `no_replay_stale_sessions` + 13 `no_baseline_scenarios` = 45, study 7,
control 0, 52 total), and the expiry now logs the literal token
`n_expired_unmeasured=N` — unconditionally, including zero, since an absent line
and a count of nought are different facts and a gate is checked by grep.

Two advisories are worth keeping as rules rather than fixes. A naive moment is
**attached** to market-local, never converted — the `_gate_moment` rule, and
`astimezone` on a naive value silently read it as machine-local. And a symbol
with **no frame at all** is `n_no_frame`, excluded from the purity fraction: it
is neither the declared source nor a fallback, and counting it as pinned would
have reported a symbol the scan never saw as evidence that the pin was working.

**A third population exists and is deliberately untouched.** The audit counted 41
records with no open and no closed scenario. Only 13 are `no_baseline_scenarios`;
**the other 28 have baseline scenarios in a status that is neither open nor
closed**, so neither expiry rule reaches them. What that status means, and
whether such a record is evidence, is a trader-and-lead question. It is written
down here so the next reader meets it as a known open question rather than as a
fresh defect.

**Answered 2026-09-06** (lead, on the trader's *"go ahead and do this yourself"*,
from the SQLite mirror read-only): all 486 baseline scenarios on the 28 records
read `status == UNTRADEABLE` - the tracker sized no position because the risk
per share was under its floor or the share count came to zero (`legacy.py`, the
`UNTRADEABLE` stamp near line 1444, the setup-level rollup near 6888). The
`tradeable` filter at the head of `build_tracker_stats_rows` already keeps them
out of every n, numerator and denominator, so nothing had to change. They are
evidence about the setup's SHAPE (a stop too tight for the standardized risk),
never about win or loss, and they are correctly not expired; 363 of the 11,372
setups carry the status. The other 13 of the 41 are `no_baseline_scenarios` and
expire as designed. The same day ruled that `EXPIRED_UNMEASURED` stays IN the
scoring population: the trader-facing exports already exclude and label it, and
a scoring population that shrank with replay staleness would let a stale week
re-rank the setup types.

### The exclusion is opt-in, because one function serves two masters

The builder shipped M3.3 excluding the expired everywhere and reported the
tension rather than hiding it; the lead ruled on the same day, and the ruling is
the rule now.

`build_tracker_setup_type_rows` is read by two callers with different rights.
`export_setup_tracker_views` renders it for the trader. But
`_load_ranked_tracker_setup_type_rows` -> `rank_tracker_setup_type_rows` ->
`apply_tracker_setup_type_adjustments` turns it into `row["score"]`, and
`tracked_setups` is the third key that ranking sorts on. Dropping records from it
is therefore a scoring change, and plan.md sec 5 forbids one without golden
fixtures first.

So `exclude_expired_unmeasured` defaults to **False**. The export passes True;
nothing on the scoring path passes anything. `export_setup_tracker_views` builds
the rows twice from one tracker — measured at 0.449 s per pass over 11,000
setups, in the after-close export and off the Qt thread — because
`payload["setup_type_stats"]` is what `_load_ranked_tracker_setup_type_rows`
falls back to and it must keep the champion's population.
`n_expired_unmeasured` is carried in both readings: "19 setups, 3 of them
unmeasured" is a fact the scoring row is entitled to state while it still counts
all 19. `build_tracker_stats_rows` and `build_band_variant_stats_rows` keep the
exclusion unconditionally, and that was checked rather than assumed — the only
thing that writes live scoring weights is `analyze_master_avwap_scoring.py`, and
it reads the ATTRIBUTE exports, which this work never touches.

`open_setups` does follow the status, and that is display-only by inspection:
`_compute_tracker_setup_type_ranking_score` reads `tracked_setups`, the metric
pair and its baselines, `target_hit_rate`, `stop_rate` and `closed_setups`, and
the sort reads `ranking_score`, `closed_setups`, `tracked_setups`,
`avg_closed_r` and `type_label`. Neither mentions it. The test asserts that
`open_setups`, `n_expired_unmeasured` and `sample_setups` are the ONLY cells that
differ, so a third one moving is a failure rather than a discovery.

**Counting the expired out of the champion's own inputs is a future
golden-fixture decision, not something to do in passing.**

One note on the test, because it took three attempts to make it capable of
failing — the 2026-09-02 lesson about tests written by the agent that wrote the
fix. The first fixture put every setup in one rank group, so `score_delta` was 0
on both sides and nothing could move. The second put the two families in
different groups, because `_tracker_priority_bucket` demotes `favorite_setup` to
`near_favorite_zone` for any family outside `MAIN_SWING_SETUP_FAMILIES`. The
third works: three families in one (side, bucket), a third family dragging the
baseline so two carry an identical positive edge, tying their `ranking_score` and
`closed_setups` so `tracked_setups` is the only separator, and `zeta_pattern`
sorting after `alpha_pattern` by `type_label` so dropping its stale records
reverses the two. `score_delta` is confidence-capped at 3 either way, so the test
asserts the rank published into `symbol_entry["priority_setup_type_rank"]` — the
value that actually moves — and asserts the delta is non-zero first so it can
never pass vacuously.

`build_recent_tracker_setup_family_rows` (the other live scoring input, which the
packet did not name) was left untouched throughout.

## Headline statistics, long form (moved verbatim from CLAUDE.md on 2026-09-03, F1 docs packet)

`CLAUDE.md` keeps the rules of this block; this is the block as it stood, with every
measurement and the reasoning behind each rule.

**Headline statistics and the priority switch (V3, decision 0016)**

- **The banner crowned max R on three and the table invented a count** (ST2,
  2026-09-06). Two defects on one screen, both reproduced through the real code
  before anything was changed.

  *The invented count.* `legacy.build_recent_tracker_setup_family_rows` computes
  `win_rate_closed` as a RECENCY-WEIGHTED mean of win flags - `exp(-ln2 *
  age_days / 14.0)` times `TRACKER_REGIME_MISMATCH_WEIGHT` on a regime mismatch -
  and the Setup Tracker's panel handed that rate to
  `swing_headline.headline_from_rate`, whose whole job is to recover the integer
  pair Wilson needs as `round(rate * n)`. That is exact when the stored rate was
  computed as `wins / n`, which is how the veto and like cohort CSVs write theirs.
  It is not how this one is written. Measured through the real writer: two
  28-day-old wins at weight .25 plus two same-day losses at weight 1.0 give
  **0.2**, and the cell printed **`25% (>=5%, n=4)`** - a 1-of-4 that never
  happened, carrying a Wilson lower bound computed from it - where the family had
  gone **2-2, 50%**. The fix is not a better reconstruction; there is no such
  thing. The builders now export `n_wins` / `n_losses` / `n_flats` /
  `n_unmeasured` / `n_pending` at each table's own grain, counted in the SAME
  loop that builds `win_flags` so the counted and the weighted readings can never
  read different episodes, and the panel uses `headline_from_counts`. The
  weighted rate stays on the table under **Win % (recency-weighted)** beside
  **Win % (unweighted)**: it is a real number that answers a different question,
  and deleting it would be the mirror of the original mistake. A row from an
  export written before the columns existed says **`counts not exported yet`**.
  `headline_from_rate` survives for its legitimate callers and its docstring now
  names them and forbids a weighted rate.

  *The crown on three examples.* `_best_now_banner_html` picked
  `max(avg_closed_r)` over any row with three closed setups, across the live AND
  study namespaces, while the table directly underneath already ranked by the
  Wilson lower bound. On the fixture that reproduces it the table lists
  `tight_and_hot` (24-6, bound 0.627) first and the banner crowned
  `fat_but_wide` (54-36 at +2.50R, bound 0.497); worse, a three-example STUDY
  with a big R could be presented as the desk's best performer, which is exactly
  the confusion between "interesting" and "working" that plan.md sec 7's
  promotion ladder exists to prevent. Both now read
  `working_lately.select_leader` on the same rows in the same order, so they
  cannot disagree. **`LEADER_MARGIN_LB` = 0.05** (five points of *lower bound*,
  not of raw rate: two rates can differ by fifteen points and still be one sample
  apart when one is thin, and the bound is the number that already knows that)
  and **`LEADER_FRESHNESS_SESSIONS` = 2** (one weekend plus a holiday; the
  tracker writes at the close slot, so a reading older than that is news about
  the desk, not about the family) were both declared 2026-09-06 BEFORE any
  forward evaluation and are not tuned to make a winner appear. Four states, each
  naming the gate that closed: `leader`, `no_clear_leader` (the reason names BOTH
  families and the gap - printing the winner of a coin flip is how a banner
  starts lying), `last_reliable_reading` (a stale input plus a `previous`
  verdict; the leader and the `as_of` are the previous one's, unchanged) and
  `no_evidence`. Lead decision the same day: `min_n` is an ARGUMENT (default
  `MIN_REPORTABLE_N`) so the two-session block passes `SHORT_TERM_MIN_SAMPLES`
  without declaring a second statistics contract, and a `no_evidence` verdict
  carries `coverage["discovery_leader"]` - the best live row that was kept out -
  which the banner prints as `No leader at the n=30 floor - leading on thin
  evidence: <side> <family> (n=12), discovery only`. **Never the word leader for
  it, and never beside a real one.** A NEW/RISING pin stays a NOVELTY badge on
  the table and is not an input to the leader.

  *The same defect, three more times.* The reviewer's NO-GO found it surviving
  wherever the packet had not looked, which is the lesson worth keeping: fixing
  the surface a defect was REPORTED on does not fix the defect. The Summary
  card's plain-English block (`research_explanations`) sat THREE LINES ABOVE the
  repaired banner still crowning `max(avg_closed_r)` on three closes across both
  namespaces - live it read *"LONG top_pattern leads at +0.99R ... 3 closes"*
  under a banner saying *"SHORT general"*, with 10 of its 17 candidates studies.
  The **Best Type Edge** tile read `setup_type_rows[0]`, so ST2.2's new
  bound-first sort silently moved it from `SHORT +23` to `LONG +14` - a tile
  that borrows another surface's ordering has no meaning of its own, and it now
  picks max `score_delta` explicitly. The Summary's **Setup types working**
  block took `rows[:8]` of that same side-first list and so showed eight LONG
  rows and no SHORT one (the first SHORT row sat at index 68 of 117); the CARD
  now picks its eight by the bound across BOTH books, with the side shown, while
  the TAB keeps side-first - a table you scroll and a card that shows eight are
  different questions asked of the same rows.

  *And twice more, in the re-review.* Both were the same shape as the three
  above: a surface computing for itself what the page had already decided. The
  plain-English card called `select_leader` directly while the banner went
  through `_remembered_verdict`, which carries a `previous` - so on a STALE
  refresh the card printed *"no clear leader. Leading on thin evidence, SHORT
  general on n=88 - discovery only"* three lines above the banner's *"SHORT
  general [last reliable reading, as of 2026-09-04]"*. Two renderers computing
  the same thing will disagree the moment one of them gains an argument, so
  `panel_verdicts(panel)` computes each horizon ONCE in `_summary_html` and both
  renderers are handed the same objects; `build_plain_english_whats_working`
  takes `verdicts=` and only computes its own when a caller has none. The second
  was a LABEL that did not come from its verdict: the 2-session block hardcoded
  "2-session discovery", which then sat over a real `leader`, and printed "the
  export carries no session, so its freshness is unstated" beside "its newest
  measured session is 58 sessions behind" - two contradictory facts in one line.
  The label is now the horizon plus `verdict_label_suffix(verdict)`, and the
  no-session sentence renders only when the gate was `no_session`.

  *"Old" is not "thin".* `discovery_basis_phrase` gives one phrase per gate -
  **leading on older evidence** for a stale row, **undated** for one with no
  session, **thin** only under the floor - used by both renderers. A row kept
  out for being old HAS the evidence; calling it thin names the wrong gate,
  which is the same class of error as calling a weighted rate a count.

  *A measured date, where the export can answer it.* Freshness was being read
  off the ENTRY session everywhere, which is right for the recent family rows -
  they carry no exit date, and a family whose newest entry is old cannot have a
  newer measured close - but wrong for the 2-session block, where it made a
  family entered eight weeks ago and MEASURED two sessions later read as 58
  sessions stale on a file written that morning.
  `legacy._short_horizon_measured_session` reads the trade date of
  `post_marks[horizon - 1]`, the same mark the R itself was computed from, and
  an episode whose marks cannot answer leaves the field EMPTY - undated, which
  reads as not fresh, never a guess. The recent rows keep entry dating until ST4
  lands `representative_exit_date`, and `LEADER_FRESHNESS_SESSIONS`' comment
  says so rather than leaving the difference to be discovered. The short-horizon
  export's identity, stated because a reader summing it wrong is the next
  defect: `n_wins + n_losses + n_flats == samples_2d`, and
  `samples_2d + n_unmeasured == tracked_setups`.

  *And once more: a renderer that has a verdict must render it.* The re-check
  found the banner's short-term block still guarded on "a discovery row OR a
  leader", falling through to a hardcoded *"not enough 2-session samples yet
  (accrues automatically each scan)"*. `no_clear_leader` matches neither
  condition, and it is the LIVE state for that horizon - twelve eligible
  families with the top two 0.001 of bound apart - so the banner said "no
  samples" three lines under a card saying "no clear leader". Both statements
  were on one screen and one of them was false. The conditional is gone. The
  general rule: a special case written beside a state machine will eventually
  contradict it, and a hardcoded sentence is a state the machine does not know
  about.

  *One clock per surface, named.* `FRESHNESS_SENTENCE` was a single constant
  saying "measured inside 2 sessions" while only the 2-session rows are
  measured-dated - so the swing line claimed a clock it does not have.
  `freshness_sentence(kind)` reads `DATING_BASIS_BY_KIND` and says
  **entry-dated** on the swing line, **measured** on the 2-session one, and
  entry-dated for any kind it does not know, which is the conservative reading
  rather than the flattering one. When ST4's `representative_exit_date` lands
  the swing entry flips to `"measured"` and the sentence follows on its own.

  *The floor is judged BEFORE the clock.* A family with three samples is under
  the floor whatever the clock says, and answering "not fresh" to three samples
  answers a question the reader did not ask. So `select_leader` splits
  `at_floor` / `under_floor` first and only dates the at-floor rows; the
  discovery pools and the reason branches run in the same order (stale, undated,
  thin), so the sentence a verdict prints always names the gate that kept out the
  row it is showing, and a row that CLEARS the floor and is merely old outranks a
  current row with three samples. `min_n` therefore binds the stale and undated
  pools by construction. **Freshness is measured on the ENTRY session** - these
  rows carry no exit date, so `latest_measured_session` is the newest scan_date
  among the episodes that produced a readable R, which is the conservative
  reading - and `FRESHNESS_SENTENCE` says so on every surface, because "fresh"
  without its clock is not a fact.

  *What was NOT changed.* `ranking_score`, `score_delta`, the Expected-R
  calibration and every pre-existing column of all THREE exports, pinned
  byte-identical by goldens taken from `main` at `84ee24d6`: the two row goldens,
  a shipped-header golden (the count columns belong at the END of the header the
  trader opens, not at the end of the inner builder - they had landed at index 26
  of 31 and 31 of 39), and `st2_short_horizon_golden.csv`. The two-session
  export gained the same additive counts plus its own `latest_measured_session`
  on the trader's answered ask, so that block is freshness-checked rather than
  discovery by construction; `win_rate_2d` keeps its value, including its
  treatment of an exactly-flat close as a zero flag, because moving it would be a
  scoring change and this packet may not make one.
- **The priority switch reorders and never withholds** - and it is **NOT BUILT
  YET** (V4 owns it; R4 B3 removed the sentence that cited a test for it). When
  it is built: "prioritise what is working" is display-only (decision 0016
  answer 5), it sorts the review queue, the M5 list and the setups table, and it
  may never hide, mute, park or withhold a row. The tier gate, movers-only and
  repetition control stay untouched by it. **The identical-visible-rows test is
  owed with the switch**, not before it - a doc that cites a test nothing runs is
  worse than a doc that says the work is owed.
- **Win rate leads every trader-facing SWING surface; MFE-after-a-held-level
  leads every DAY-TRADE surface** (decision 0016 answers 3 and 4). The trader
  gives swings room and their losses run ~1.5x their best wins, so mean R ranks
  their swings by the statistic their loss profile makes misleading. Win rate
  goes FIRST, with `n` and a **Wilson lower bound** beside it (`swing_headline`),
  and **sorting is by the lower bound** - the raw rate puts a 100%-on-three cell
  above a 62%-on-ninety every time. Mean R stays beside it, never replaced.
  **PARTIAL** (R4 B3). Wired: the AWAY digest's swing ranking (A11), the setup
  docs' record line (`setup_docs.family_record_sentence`, rendered at read time
  from the tracker), the Master AVWAP setups table's **Family Win %** column, the
  Setup Tracker's **Last 30 Days** tab, and all four Weekend Prep cohort tables
  (veto, like, pass, rejection), which now sort by the bound. **Still owed: the
  Setup Tracker's Setup Types tab**, and the reason is measured rather than
  scheduling - `master_avwap_setup_type_stats.csv` carries no win column at all
  (only `target_hit_rate` and `stop_rate`, which are different questions), and
  the outcomes file cannot be joined at that table's grain: its 184 rows collapse
  to 71 (side, bucket, family, zone) groups, so a joined rate would repeat across
  up to six rows and read as each row's own. **ONE WILSON**: `swing_headline`'s z
  (1.96, 95% two-sided) is every trader-facing win rate.
  `master_avwap_lib/expected_r.py`'s z of 1.28 is a PARAMETER of the Expected-R
  proven-quality score inside a fenced scoring file, not a column anyone reads;
  no trader-facing surface may reach for it. On
  the day-trade side the headline is `held_run_score`: P(the level held in the
  first 30 min) x trimmed-mean MFE_R of the ones that held. **ONE formula reaches
  every surface** (R4 A10): the Day Trade Tracker joins
  `held_run_score.dimension_summaries` and computes nothing, and the M5 alert row
  reads `alert_cell` + `alert_suffix`. **The join is an equality, so this module
  spells its segments the AGGREGATOR'S way** (R4 fix round 1) - the champion's own
  `time_bucket_for`, an episode counted under EACH of its bounce types, and the
  combination `+`-joined. Four of the tracker's nine tabs fill (`bounce_type`
  36/36 live rows, `bounce_combo` 58/59, `time_bucket` 10/10,
  `market_environment` 10/10); the four `master_avwap_*` tabs are BLANK because
  the outcome log does not carry them at all, and `rrs_alignment` is blank because
  it is REACHABLE and not derived yet - `held_run_score.UNDERIVED_DIMENSIONS`
  keeps those two facts apart. A blank is right where the question cannot be
  asked; a second formula under the headline key is worse than a blank when the
  column is read as an ordering, and a spelling that silently blanks a tab the
  data CAN answer is worse than both. `d1_setup_present` is fed from the
  scanner's own `master_avwap_tracker_scoring_snapshot.json` (19 MB), never from
  the 1.1 GB setup tracker, and its index **expires on the day roll** - a memo
  that never rolls puts `d1_setup_present` back to False on day 2 of uptime and
  stops being "lately" while still saying it is. **Every number on that table
  names its own basis** (R4 B4): the champion tier is a COLUMN (PROVEN / MUTED /
  active from the bounce learning state, blank for a segment it never saw - live
  4 / 2 / 185 / 104 of 295 rows), and the aggregator's verdict is headed
  **"Verdict (edge score)"** because it is computed from average R and sits three
  columns from a headline computed from something else. **The My Decisions tabs
  carry the headline too**, through the same `apply_held_and_ran`; those rows name
  no side, so `held_run_score.ALL_DIRECTIONS` gives them a pooled cell
  accumulated FROM THE EPISODES - never an average of the long cell and the short
  one, which would be a mean of trimmed means and a second formula in that file
  again.
- **The AWAY digest ranks swing picks by the tracker's record, not by the bucket**
  (V1 item 3, built R4 A11; decision 0016 answer 8: *"the best pick is often in
  the near bucket, not the favourite bucket, so the cream is not being sent."*)
  The order is the **Wilson lower bound** on the setup family's realized win rate
  - `master_avwap_tier_outcomes.csv`'s own `win` column inside `lately_window()`
  - at ONE DECLARED HORIZON (`evidence_stats.SWING_HORIZON_SESSIONS`, 5, which
  `autopilot_core.SWING_DIGEST_HORIZON_SESSIONS` re-exports and `setup_docs`
  reads too - R4 B2),
  with expected R as the tiebreak; an ungraded family sorts BELOW every graded
  one rather than at zero. **The horizon is declared because that file is one row
  per (pick, horizon)**: pooling all four inflated n ~2.5x with correlated looks
  at one decision, which tightens every Wilson bound unevenly and CHANGES THE
  ORDER. A row the tracker flagged `stale_horizon` is dropped, the way the
  scan-factor leaderboard already drops it from the same file. The bucket is PRINTED and never ranked on, and the near
  cap is applied **after** the ranking, so what is hidden is the weakest near rows
  and never the best one. The read is the caller's, so `render_away_report` stays
  a pure renderer. AWAY is still the only routine pusher.
- **The Research tab is not a trader surface.** It is the builder's
  (decision 0016 answer 7: the trader never opens it). Nothing the trader must
  see may live only there - a number that matters gets a line on the Trading
  Desk, the Journal, Weekend Prep or the AWAY Recap, and the full readout stays
  in Research. The same rule retires "it is on the Research tab" as an answer to
  "where does the trader see this?"
- **"Lately" is ONE number and it is counted in trading sessions.**
  `evidence_stats.LATELY_SESSIONS` (20) is the home; `lately_window()` walks the
  exchange calendar. Twenty calendar days is fourteen sessions in a normal month
  and twelve across a holiday week, so a calendar window silently shortens the
  sample exactly when the market was closed. **The review board is inside this
  rule** (R4 B6): `review_learning.DEFAULT_WINDOW_SESSIONS` IS `LATELY_SESSIONS`,
  it was a 90-CALENDAR-DAY literal, and the blind-spot and leak callouts are cut
  on it. Weekend Prep's week is `evidence_stats.WEEK_SESSIONS` (5) for the same
  reason - it printed "Week of <Mon> to <Fri>" over the last 7 calendar days, so
  a holiday week measured four sessions and still called itself a week. The state
  key, the report header, the CLI flag and every renderer say **sessions**.

### Held is MEASURED held (packet Q1, 2026-09-04)

Process review 2026-09-04, findings 1 and 2. `Episode.held` was `not broke_early`, so an
episode nothing had followed up read as held. Recon on the live outcome log (default
window, 8,161 episodes): 2 registered-only, 977 with rows but none reaching 30 minutes,
1,960 broken inside 30 min, 5,222 held past 30 min - and all 979 unanswered ones read
`held=True`. The producer (`legacy.py` `BOUNCE_OUTCOME_COLUMNS`) writes `stop_hit` as a
boolean over ALL bars since entry and no first-break time; `minutes_elapsed` is entry to
the LAST bar the row knew. A `final` row is written only by the per-symbol update path or
the sweep, whose autorun is OFF by default, which is why registered-only events exist.

Rule now: `measured_held` / `measured_broken` / `pending` / `unmeasured` per episode, only
the first is held, `hold_rate` = held / measured, counts and `coverage` on every cell and
a Measured column on the tracker. A stop first seen past the window with no earlier
no-stop row at or past 30 minutes is `break_time_unknown`. Two producer changes are OWED
and ask-first: a `stop_hit_at` column, and the sweep autorun default.

D1 overlap: the scoring snapshot carries `side` per setup and `scan_date` only (no time),
and the join dropped the side - 8 of 2,646 "D1 present" episodes were the opposite side.
The join now keeps it (`aligned` / `opposed` / `none` / `unknown`), only ALIGNED carries
the privilege, a missing snapshot is UNKNOWN, and every summary carries
`d1_basis: same_session_retrospective` because "known when the alert fired" cannot be
established from a date-only file. "Lately" is `evidence_stats.lately_window` with
`window_report` naming the missing sessions; the old "last 20 dates present" widened
silently on sparse data.

## Frozen exe rebuild policy, long form (moved verbatim from CLAUDE.md on 2026-09-03, F1 docs packet)

`CLAUDE.md` keeps the policy, the guards and the triggers; this is the section as it
stood, with the Smart App Control history, the `d0aebd5` delivery-gap story and the
selftest count history.

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
    `ops` are bundled. Phase 0.31 made one `ai_jobs` module reachable from Trade Mentor, so the
    spec collects only `ai_jobs.market_story_narration`, not the source-only nightly tree.
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
  - **Codex build-host DLL fence (2026-09-15).** A Codex desktop shell adds its own poppler and
    libheif directories to the DLL search path. PyInstaller once copied their ICU, private CRT
    and OpenSSL files beside the desk: Qt failed with "procedure not found", then SSL vanished
    when the first broad filter removed too much. The spec now pins build-time qtpy to PySide6,
    excludes binaries whose source is the Codex runtime cache, and replaces an intercepted
    OpenSSL pair with the current Python runtime's own DLLs. The frozen selftest is 86/86.
- **Triggers — a change of these kinds can break the bundle, so rebuild and run the frozen selftest:**
  1. New third-party dependency (`requirements-*.txt` / `constraints.txt`) — may need hiddenimports or `collect_data_files`. **Not** covered by the guards.
  2. New non-`.py` runtime asset. The spec mirrors every `FIRST_PARTY_PACKAGES` tree plus `config/`; an asset outside those silently goes missing. *(spec-drift test catches it)*
  3. New top-level package under `scripts/` that is imported lazily — the spec's `collect_submodules` list is hardcoded. *(spec-drift test catches it)*
  4. New dynamic import by string name (`importlib`, name-keyed panel/service lookup) in an uncollected package. *(add the module to `selftest.LAZY_ENGINE_MODULES` — but only if a frozen run can actually reach it; see the disjointness rule above)*
  5. Any change touching `__file__` / `ROOT_DIR` / `sys.path` — `ROOT_DIR` is `sys._MEIPASS` when frozen. **Not** fully covered; the selftest checks the phantom-root assumption only.
- Read `packaging/README.md` "Things that will bite you" before touching the spec or any of the above.
  The signature failure is a bundle that starts fine and dies at the first lazy import, so "it launched"
  is not proof; the selftest is what exercises the engines.

## Core loop rules, long form as of 2026-09-05 (moved verbatim from CLAUDE.md, repo cleanup)

On 2026-09-05 `CLAUDE.md`'s "Core loop / data flow" section was 54.8 KB of an 78.6 KB file that loads into every session - it had regrown 13 KB since the 2026-08-28 split because new rules were written into it long-form. Every bullet below is the text as it stood at `e7b12ebe`, unchanged; `CLAUDE.md` now carries each rule in one to three sentences and points here. Where a bullet already has its own entry above (Phase 0.12, P1, P9, P10, R4 Part B, F1, T1, Q4, the tee, the headline statistics), this is the second copy and the entry above is the fuller one.


Each rule below is binding as written. The incident, measurements and trader
conversation behind every one are preserved verbatim in
[`docs/DESK_INTERNALS.md`](docs/DESK_INTERNALS.md) — **read the matching entry there
before changing the behaviour a rule governs.**

**Shape**
- Entry: `launch_gui.py` → `scripts/ui/app.py` (PySide6 Trading Desk). One desk role, no flag to change it — Desk Link/satellite and the mini-PC scanner were retired 2026-08-08 and their code removed 2026-08-24 (no `desk_link`, no `ui/satellite.py`, no `master_avwap_mini_pc.py`, no `--satellite`/`--desk-role`). **The legacy Tk UI, its shims, the Tk journal/market-prep tabs and `TickerMover.py` were REMOVED on 2026-09-03** (assessment packet F2); there is no second UI and no `PyQt5` in the dependency set.
- Market data: IBKR TWS/Gateway `127.0.0.1:7496` (`ibapi`) primary, `yfinance` fallback; bar source tracked per scan (`docs/BROKER_ADAPTERS.md`). **On the desk the D1 scan's daily bars are PINNED to Yahoo** by `local_settings.json` `daily_bars_source: "yahoo"` (R10.0b §1.3 interim pin, `master_avwap_lib.daily_bars_source_pin`) - a run manifest showing 400+ Yahoo daily-bar successes and single-digit IB is the pin working, not IB failing (checked 2026-09-03). IB serves the intraday bars and the champion's M5 loop.
- Engines: `scripts/master_avwap.py` (+`master_avwap_lib/`) D1 AVWAP swing scanner; `scripts/bounce_bot.py` (+`bounce_bot_lib/`) intraday M5 bounce detector; `market_prep/` pre-session services.
- Inputs: plain-text watchlists (`longs.txt`, `shorts.txt`, `swinglongs.txt`, `shortswings.txt`) in the shared home folder.
- Storage: home folder `C:\TradingBotData` is a plain LOCAL folder — **there is no cloud drive** (removed 2026-08-10, decision 0015). Per-machine caches/diagnostics under `%LOCALAPPDATA%\TradingBotV3` (`scripts/project_paths.py`). The DAS `\\MINI-PC\Trading Bot Data` is the durable tier; **write local first, move to the DAS after**, so an outage costs throughput and never correctness.
- Shadow engines (`market_state.py`, `greatness_monitor`) emit JSONL promotion evidence only. Review-learning loop: Alert Center decisions → `alert_review_events.jsonl` → `review_learning.py` → AI-curated `review_policy.json` → chart annotations (`docs/REVIEW_LEARNING_LOOP.md`).

**Research warehouse** (Phases 0–8 built; contract `docs/ULTIMATE_SETUP_DATABASE_PLAN.md`, decisions `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`, identities `docs/RESEARCH_WAREHOUSE_ERD.md`)
- Shadow-only additive evidence: **zero detector/score/alert influence.** Lives at `research_store_dir`, a separate storage class NEVER inside `C:\TradingBotData` (unset = disabled).
- **The post-scan build runs in an owned child process (F1, BD-95); reads remain session-scoped.** Partitions are MONTH-keyed: narrow through `ResearchStore.read_rows` (Arrow-side `symbols` / `interval_start_range`), never by filtering a materialised list.
- **Never widen `_run_outcomes` to a date filter** — its walk runs FORWARD across sessions (BD-66/BD-69/BD-74).
- **The seal de-duplicates at the dataset grain; nothing upstream is trusted to have done it** (BD-96). `bar_m5` for August 2026 was 85% duplicate rows because the tee's dedupe state reset on the UTC date and `seal_spool` published what it was given. `SealResult.rows_deduplicated` is the count; `SUPERSEDING_DATASETS` are exempt. Repair is `research_warehouse.cli dedupe --apply` - a COMPACT-shaped rewrite that keeps the earliest `observed_at`, retires its inputs and writes `rows_dropped` on the manifest line - and **the derived bars and intraday features computed from a duplicated month are wrong in VALUE, not duplicated**, so they need a rebuild after the dedupe, not a dedupe of their own.
- A SNAPSHOT over 64 MB is stored whole but **never `json.loads`-ed**; answer the UNCHANGED watermark from a chunked hash before any `read_bytes` (BD-73).
- Growth is month-keyed: it worsens all month and resets on the 1st. Check the calendar before treating a new report as new.
- **H2 exists again** (`TIMEFRAME_MINUTES`, `DERIVED_TIMEFRAMES`, BD-78): the locked plan cut it for having no consumer and the Phase 0.12 higher-timeframe LRSI study is one. RTH is 6.5 h, so H2/H4 end each session with a STUB - published as evidence, **excluded from the LRSI input**.
- **The HTF LRSI grid is 16 diagnostic recipes and never a Cartesian search** (`outcomes.HTF_LRSI_RECIPES`): 4 timeframes × 4 entries, one stop model, one target. Long and short legs read the **SAME unmirrored series** - cross-up 50/20, cross-down 50/80 - because the formula clamps at 0 and the mirrored-close idiom is a DIFFERENT feature (BD-79). Live `CROSS_LEVELS` stays `(20, 50)`; `RESEARCH_CROSS_LEVELS` is additive and shadow-only.
- **`anchor_instance` comes from `earnings_avwap_anchors.csv` and the SCAN feeds that CSV** (2026-09-04). The warehouse reads earnings anchors from the bronze wrap of that one file (`cli.anchors_from_bronze`) and nowhere else; until 2026-09-04 it held 14 hand-imported rows, so the swing bands were 99% null and `swing_house_v1` graded 0/257. `runner.bridge_earnings_anchor_caches_to_csv` now appends every symbol's cached current and previous earnings anchor after each scan through `append_anchor_candidates` - append-only, de-duplicated on (ticker, anchor_date), new rows at the END (bronze's watermark is a line offset), a failure logged and never raised. **Nothing live reads the CSV**; `run_anchor_watchlist_scan` is its only other reader and has NO caller - never wire one up without noting it would fetch IB bars per row. Never trim or "clean up" that CSV: the newest two distinct dates per ticker are the current and previous anchors, and older ones are history.
- **A reconstructed anchor is LABELLED and is never promotion evidence, and the swing outcome row names its path** (packet Q2, 2026-09-04, BD-99/BD-100). `anchor_dates_by_symbol` returns an `AnchorChoice` stamped `observed` or `reconstructed` from the anchor row's own `system_from` read market-local; `feature_snapshot_daily.anchor_knowledge` carries it and a NULL reads as `legacy`, never as observed. `outcome_path.path_kind` says `managed` / `plain_target` / `plain_no_target` and is excluded from BD-98's unchanged-comparison so no stored row is rewritten just to gain a label. `cli band-coverage` reports coverage read-only per recipe and bucket; `cli rebuild-daily-features` (dry run by default) recomputes past sessions with their anchors, and the repair order is build -> rebuild-daily-features -> recompute-outcomes -> band-coverage.
- **The setup registry is frozen DATA and is not authoritative yet** (P7, 2026-09-01). `scripts/setup_registry.py` loads `setup_registry_v1.json` - one entry per setup, keyed `setup_id@version` - joining the FIVE places that name a setup: `_FAMILY_TAGS` (the canonical id), `setup_docs`, the playbook study, the claim picklist, and `legacy.py`'s `*_STUDY_FAMILY` constants (eight families are named ONLY there). Regenerate with `scripts/build_setup_registry.py --write` and review the DIFF; it is never rebuilt at import. It **resolves no disagreement** - those are `known_divergences` - and **fills no column its sources do not establish**, so supported sides and timeframe roles are deliberately blank. An unknown name RAISES rather than defaulting to `GENERAL`. **`plan.md P4.1` is where it becomes authoritative**; until then nothing in production imports it. Its sibling `research_warehouse/trial_ledger.py` records one append-only row per registered grid BEFORE any outcome is inspected, and `register` refuses to rewrite an existing `trial_id`.
- **The like-link payload field is `match_basis`, and `LikeLink.from_payload` is its only reader** (Q3.4, 2026-09-04). Both lake audit scripts had reached for a `basis` key with an `"unknown"` default, so every link on the lake read as unknown; `from_payload` is strict in BOTH directions (a missing key and an unknown key each raise, naming it), `basis_of` and `count_payload_bases` read through it, and an unreadable payload is an audit error that prints the offending row rather than filling a bucket. The dataset is `bronze_like_occurrence_link` - the unprefixed name reads nothing.

**Alert Center, review queue and capture**
- **The charts own the review pane**; between them and the tab strip there is at most ONE slim row (the verb row). **The arm bar stays UNDER the chart** — placement is a HOST decision via `AlertChartReview(dock_arm_bar=…, dock_capture_rail=…)`. **Never propose moving the arm bar without asking.** `CaptureRail.action_shortcuts()` is rebound at panel scope; a `QShortcut` inside a hidden tab never fires, and two bindings for one sequence fire NEITHER.
- **A VETO retires the chart and a CLAIMED like ADVANCES it (an advance parks nothing and drops nothing); a QUICK like and a NOTE move nothing** (trader, 2026-09-04, second pass: *'a double click of any of the setups there should be sufficient ... double clicking that box should advance the chart'*; first pass, for the quick like: *'I still need time to enter alerts'*). A rail veto takes its OWN verb (`vetoRetireRequested` → `_retire_after_veto`), never the "✕ Not today" button's: **that button writes an uncoded second row and opens the note box, and the rail's veto opens neither and writes ONE row.** Both retirements are one body with a flag, so the auto-pick / faded / Focus-review branches cannot drift apart. A QUICK like is REPORTED (`likeRecorded` → `_after_like`) and moves nothing; a CLAIMED like ADVANCES (`likeAdvanceRequested` → `_advance_after_like` → `_advance_review_queue`), and an advance is not a retirement - nothing is parked, no Focus pick is dropped, the symbol's other queued alerts keep their places. `like_mode_of` decides which, absence reading as claimed. BOTH write the review event `like_advance` through ONE helper (`_record_like_advance`) because `review_learning.TAKE_ACTIONS` keys on that string. "Veto D1 — but M5 today" writes a veto row and emits a REQUEST; the panel performs the Focus placement, so the Focus store keeps one writer. Place first, retire second (through the box-free verb, lead ruling 2026-09-04); a failed placement still retires. What a LIKE offers is `MAIN_CLAIM_GROUP` + `EXTRA_CLAIM_IDS` in `ui/annotations/setup_claims.py`; keys run in list order so learned digits never move.
- **A day-trade PASS is a note, not a veto, and it never retires the chart** (trader, 2026-08-31: *"I really like this stock for a daytrade but it has this ONE issue"*). Multi-select reason codes come from a SEPARATE vocabulary family, `ui/annotations/vocabularies/pass_reasons_v*.json` — never folded into the veto list, whose cohorts are already accruing forward returns. Codes are written in VOCABULARY order, not click order. When the desk already holds the symbol's M5 bars the row references one session of them through a sidecar (`ui/annotations/pass_bars.py`, written BEFORE the row so a reference never lies); with nothing cached the row still writes, timestamp only. **A capture click never fetches.** A pass does **not** mark the symbol "Reviewed today" - `pick_feedback._ANNOTATION_DECISIONS` stays `veto`/`like_claim`/`note`, because that set is read by the scanner report and several badges (trader decision, 2026-08-31; both this and "never retires" are DECIDED, not open).
- **A LIKE has two modes and only one of them names a setup** (P9, 2026-09-02, trader: *"anytime I like and claim a setup or like a day trade setup I just want to let the bot and the future AI know 'something about this was good'"*). **Alt+L** writes a QUICK like - `like_mode: "quick"`, no claim, no why - and **Alt+K** the claimed one, which needs only the claim - the why is optional since 2026-09-04 (packet T2), and a double-click on a setup is the whole gesture. **The key is instant and the BUTTON prompts**: the chart's verb row and the rail both carry a quick-like button that opens a box for an OPTIONAL note (cancel records nothing), while Alt+L never prompts - a key that stops to ask is not a one-key verb. An optional note is not R9.2(a)'s required why returning: that rule is about a CLAIM. R9.2(a)'s why-required is superseded for the quick path (P9) and, since 2026-09-04, for the CLAIMED path too. A quick like LEAVES THE CHART UP (2026-09-04; it retired until then, and a CLAIMED like advances again since that day's second pass), records `like_advance` and marks the symbol reviewed exactly as a claimed one does, and **places nothing** - a like carries zero privileges (plan.md P3.1). It grades under `like_unclaimed` and contributes a **LINK** to the auto-tagger, never a tag, because it names no setup. `like_mode` is ADDITIVE (schema stays 1) and its ABSENCE reads as `claimed` - a claim was required until P9. **A capture sidecar is completed after the close** by the `sidecar_completion` slot into a NEW file (`m5_bars_completed_ref`); the original `m5_bars_ref` still means "what the desk held at the click" and is never rewritten.
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
- **The post-scan warehouse build runs in a CHILD PROCESS at below-normal priority, never a thread**: a CPU-bound Python thread holds the GIL (83% measured 2026-09-03) and no priority or timer trick frees the GUI. The exchange calendar is memoized. The stall watchdog's cap is per HOUR.
- **The research M5 tee de-duplicates BEFORE it does any per-bar work, and its mark is persisted and never reset by a clock** (BD-96, 2026-09-03 evening: the tee thread was 101% of one core and 91% of GIL samples, five hours after the close, because every 60 s it parsed, session-tagged and hashed all 346k cached bars and THEN dropped them as duplicates). `capture_m5_tee` is two passes - identity first, work for survivors only - and the live desk's state is a per-symbol high-water mark in `tee_high_water.json` beside the spool; a symbol whose newest bar is behind its mark is not walked. **The seal de-duplicates at the dataset grain and counts what it drops** - the old `seen` set reset on the UTC date and put 10.2M duplicate rows into `bar_m5` for August; `research_warehouse.cli dedupe --apply` (dry run by default) repairs a partition with a COMPACT-shaped rewrite. **Every thread's CPU time is measured once a minute** (`ui/thread_cpu_gauge.py`, always on): the stall watchdog names only stalls the GUI thread caused, and a thread holding the lock leaves the GUI stack innocent (816 s attributed to `app.exec` that day). **A recon that rates a timer from its docstring has not measured it.**
- **The daily pick scorecard streams both CSVs on its own worker and writes `picks_scored_at` only on success** (packet Q5, 2026-09-04: 15.7 s on the Qt thread the day before). `_maybe_score_picks_daily` decides and starts `autopilot-scorecard`; the wrap-up worker runs the body inline through the same guard; `autopilot_core.read_scorecard_inputs` keeps today's rows only; a failure keeps the last-good line, raises for retry, and gives up for the day after three attempts (`picks_scoring_failed_at`).

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
- **The setup tracker is mirrored into a SQLite record store after every JSON save, and the JSON is still the truth** (decision 0017, F3 step 1, 2026-09-04). `scripts/tracker_store.py` writes one row per record with a content hash, behind `tracker_storage_shadow` (default ON), after `save_json` and never able to fail the save; no reader may load from the SQLite until gate #57 (five parity-clean live saves, `python scripts/tracker_store.py verify`) is met, and then readers move ONE AT A TIME, narrowest first. **A terminal outcome row is re-simulated only with `force`** (BD-98): the nightly never passes it; `cli recompute-outcomes` does, one lock per bucket, writing only where the result changed.
- **Address home-folder stores by their `project_paths` named constants** — resolving by name under the wrong root shipped a blank page for six days.
- **The overnight runner's `veto_cohort_grading` slot is deterministic and calls no model.** **The order is decision 0018's: deterministic slots, then the digest, then narration, then the model-gated slots; a later phase appends inside its stage and never reorders across stages** (`docs/decisions/0018-deterministic-stage-before-narration.md`, 2026-09-04 - the original two narration slots held up to 2½ h of reserve ahead of every deterministic slot, a slot that cannot fit its reserve records SKIPPED, the 2026-09-01 run took six hours, and no deterministic slot reads a narration slot's OUTPUT). Phase 0.31 appends `market_story_narration` at that stage's end. The order is pinned once, as `EXPECTED_SLOT_ORDER` in `tests/test_ai_jobs_runner.py`. Nothing in this chain may reach a detector, score, alert, watchlist, Focus, the review queue or `review_policy.json`.
- **The digest gate has TWO halves and both are measured** (packet Q4, 2026-09-04). `clean_digest_sessions` counts a RUN of CONSECUTIVE clean exchange sessions ending at the newest pack, walked through `market_calendar.previous_session` and **never by weekday arithmetic**: clean is `is_session` plus an EMPTY `unavailable` (the pack's own failure record - its summary already calls such a pack INCOMPLETE), a non-session pack neither counts nor breaks, and `first_gap_session` names where the run stopped. `sessions_collected` keeps its pre-Q4 distinct-count meaning for existing readers. The second half is a FILE: `digest_audit_approval.json` beside the packs, written **only** by `python -m ai_jobs.digest approve-audit --pack <date> …` (run from `scripts/`; refuses fewer than three packs and any date with no pack), and **no nightly job may write it** - a runner that approves its own evidence has asserted, not audited. `gate_met = window_met and audit_recorded`, and **`journal_enrichment` refuses until both are true**, its ledger row reading `refused: audit not recorded`. `review_policy_draft` and `setup_research` have their OWN gates and are untouched by this. **The System Health gate strip shows the number the gate TURNS ON** - `sessions_consecutive_clean`, not `sessions_collected` - and the Enrichment counter's `met` reads `gate_met`, so the strip can never say "Digest 11/10" at a two-session run or "Enrichment met" on a night the slot refuses.
- **`entry_index.json` is the compact handoff and it is deterministic** (Q4.4). Written beside the packs at the end of `run_daily_digest` with a temp-and-rename, where a failure is logged and **never fails the digest**. Four sections that are **never merged** - `intraday_held_run` (MFE/MAE only; `close_r` is the RESULT and is not blended in), `swing_win_rates` and `journal_execution` (both EMPTY BY CONSTRUCTION with the reason stated, because the fact pack carries champion INTRADAY outcomes and no journal block), `preference_observations` (classified by `review_learning`'s TAKE/REJECT sets, not by a second list) - plus `changes_vs_prior_window` **by FLOOR STATUS only, never by ranking immature cells**, carrying `this_window_packs` / `prior_window_packs` so "46 cleared, 0 fell" cannot read as a finding when the prior window simply had no packs, the registered trials listed UNRANKED with their frozen windows, and an `open_questions_for_a_ticker_brief` that stays empty because a brief opens only for a STATED question. **Every `pack_path` it prints is the file the numbers were READ from** - the newest superseding sibling from `latest_pack_files_by_session`, never `facts_path`'s version 1, because a citation to a superseded pack points the reader at the record that was corrected (3 of 9 live sessions are superseded); a same-second tie breaks on the supersession index, never on the file NAME, which sorts `.1` before the base file. `repo_commit()` resolves HEAD through the `gitdir:` pointer as well as a plain `.git` directory, because in a git WORKTREE - where every agent builds - `.git` is a FILE and the manifest's own helper returns "".

**Charts and boards**
- D1 charts carry a volume underlay and an earnings ribbon drawn INSIDE the price view; neither votes on the price range. Earnings headroom is reserved for EVERY symbol. The cache holds no future dates, so the next report is projected and labelled `est`. Payloads are built on the ChartDataService worker, never on the paint path.
- Chart paint lines: `scripts/chart_levels.py` builds the `levels` payload on the **worker**, never the paint path.
- Price alerts: Focus and Research share one `PriceAlertService`. The `read_only` mode survives the Desk Link removal and now has no production caller.
- **The group RS/RW tape owns its own clock** (`scripts/group_rrs.py` pure formula + `ui/services/group_tape_service.py`): ONE batched `yfinance` download per 5-minute tick, no retry inside the tick, **zero IB traffic and no `legacy.py` change**. Session filter is completed M5 bars **plus a same-date filter**; a window without `length + 2` bars is `None` and draws NOTHING (0.0 would claim "in line with SPY"). The RS Window tab still reads `rrsSnapshotChanged` — it answers a different question.
- **M5 Strength Board:** batched yfinance over `universe_all.txt` **PLUS the four trader watchlists** (V1, decision 0016 answer 9 - a name the trader follows may not clear the universe's liquidity spec, and the board it never appears on is the one they read), **zero IB traffic**. Its relative volume is **SESSION-RELATIVE** (R4 A7): bar k of today against bar k of each of the prior 15 sessions, and a session that never reached bar k **contributes nothing rather than a zero**. The D1 SMA floors read a **`2y`** daily download with **today's forming bar dropped** (R4 A8). `relative_volume` is deliberately NOT one of the seven fenced formula functions; those seven stay byte-identical to the R8 baseline. Every board add re-runs the adoption gate at click time. **Its parity rows also auto-join M5 Focus** (trader, 2026-09-04: *"I want all shorts and longs on the RS/RW board TC2000 to bne auto added to the M5 focus picks"*): `_auto_adopt_strength_board` runs on `boardChanged` and once at attach, over rows with an EMPTY `failed_floors` only (a greyed near-miss is never adopted), re-running the ONE adoption gate on each row's own numbers, skipping any symbol in `_ignored_symbols` **and any symbol the trader took OFF a focus side today through ANY door** (`FocusPickStore.declined_today`, recorded by the STORE on every removal - `remove`, `remove_everywhere`, `clear`, the fade - under an additive `declined` key in `focus_auto_picks.json`, same-session only and pruned on load; `_ignored_symbols` alone holds only what the "Not today" verb parked, so the Focus-review walkthrough, the Focus list's remove button, the cross-focus toggle and the Master AVWAP unfavorite were each undone by the next refresh), **DESK only**, and writing through the STORE plus `mark_auto_adopted` - **never `FocusService.add`**, which would forge a trader "like" into `pick_feedback`. It **never removes** (the ten-session fade and "Not today" own removal) and an existing entry is counted, never re-marked. Adds are BATCHED one `add_many` per side (60 names one at a time measured 781 ms on the Qt thread); the marker stays per name. **Every adopted name is also injected into the shared `longs.txt` / `shorts.txt`** by `FocusPickStore._inject_into_shared`, as every Focus add always has been - so this auto-join grows BounceBot's intraday scan input, and a removal un-injects it again. **A row click charts into the Visual Alert Review pane** via `chart_symbol` (the lookup box's door), never `_enqueue_review_alert` (the scanner's door, which drops in AWAY, drops parked symbols, diverts M5 to the alert bar and can hide a row behind movers-only). **Since 2026-09-03 that is the rule for EVERY ticker click on the Trading Desk** (trader: *"the main tab should always be centralized with the main chart"*): the RS/RW, entry and Focus-strength boards inside the Alert Center chart through `_chart_board_symbol` unconditionally, a feed ticker-name click charts the alert itself, and the setups column's four panels (setups table, RS Window, Industry Board, Watchlists) carry a `set_chart_sink` that `TradingDeskPanel.set_mode` points at `chart_symbol` in workspace mode and clears in tabs mode, where the pane is on another tab and the popup is right. **A board chart holds NO place in the waiting list and is never re-queued or skip-counted** (trader, 2026-09-04: *"once i look and click off, its done"*): `_is_manual_chart_look` is the exact `MANUAL_CHART_TAG` test, a look belongs in no P(take | shown) denominator, and clicking away from one writes NOTHING. **Looking at a name that was WAITING takes it out of the waiting list and it does not come back** - that IS *"once i look and click off, its done"*, and it is the one case where a board look changes the queue at all. The M5-alert-bar `skip` with `clicked_away_from_m5_alert` is a different population and is untouched, and a dequeued D1 chart still returns to the head. The snapshot popup survives as `show_board_symbol`, the door for a board on ANOTHER page (the AWAY Recap), and as the standalone-panel default (`None` sink). **It is a section under the Desk's Strength window, not a page** (trader, 2026-08-31) - one `StrengthBoardService` owned by `MainWindow`, hosted through `AlertCenterPanel.attach_strength_board`, **starting closed** because the alert column's 360 px floor is width the charts would otherwise lose. **The RS/RW board is a SECTION in that same column and no longer a tab** (V1, decision 0016 answer 7): it sits ABOVE the M5 Strength section, **starts OPEN** while Strength starts closed, and is hosted in a scroll area because bare it took the column's floor from 190 px to 452. The RS Window tab still reads `rrsSnapshotChanged` - it answers a different question.
- **One completed-bar rule** (`scripts/completed_bars.py`): `bar_start + bar_minutes <= now`, **inclusive**, timezone-converted with `astimezone` and **never** `replace(tzinfo=None)`. BounceBot's ad-hoc copies migrate opportunistically, never as a silent change to a shipped detector.
- Pure indicator modules (`scripts/indicators/`): completed bars in, immutable tuples out, `None` for anything unmeasurable. **No importer yet — the first one fires the packaging trigger.**
- Auto/Away phone output: `autopilot_today.txt` is the single verified home-folder digest, safety/freshness header first.

**Headline statistics and the priority switch (V3, decision 0016)** — long form with every
measurement in `docs/DESK_INTERNALS.md` ("Headline statistics, long form").

- **The priority switch reorders and never withholds - and it is NOT BUILT YET** (V4 owns it).
  When built: display-only (decision 0016 answer 5); it sorts the review queue, the M5 list and
  the setups table, and may never hide, mute, park or withhold a row. The tier gate, movers-only
  and repetition control stay untouched by it. **The identical-visible-rows test is owed with the
  switch**, not before it.
- **Win rate leads every trader-facing SWING surface; MFE-after-a-held-level leads every
  DAY-TRADE surface** (answers 3 and 4: the trader's swing losses run ~1.5x their best wins, so
  mean R ranks their swings by the misleading statistic). Win rate goes FIRST with `n` and a
  **Wilson lower bound** (`swing_headline`), **sorting is by the lower bound**, mean R stays
  beside it. **PARTIAL**: wired on the AWAY digest's swing ranking, `setup_docs`'s record line,
  the Master AVWAP **Family Win %** column, the Setup Tracker's **Last 30 Days** tab and all four
  Weekend Prep cohort tables; **still owed: the Setup Tracker's Setup Types tab**, because
  `master_avwap_setup_type_stats.csv` carries no win column and the outcomes file cannot be joined
  at that table's grain (a joined rate would repeat across rows and read as each row's own).
  **ONE WILSON**: `swing_headline`'s z (1.96) is every trader-facing win rate;
  `master_avwap_lib/expected_r.py`'s z of 1.28 is a PARAMETER inside a fenced scoring file and no
  trader-facing surface may reach for it.
- **The day-trade headline is `held_run_score`**: P(the level held in the first 30 min) x
  trimmed-mean MFE_R of the ones that held. **ONE formula reaches every surface**: the Day Trade
  Tracker joins `held_run_score.dimension_summaries` and computes nothing; the M5 alert row reads
  `alert_cell` + `alert_suffix`. **The join is an equality, so the module spells its segments the
  AGGREGATOR'S way** (the champion's `time_bucket_for`, an episode under EACH bounce type,
  combinations `+`-joined). The four `master_avwap_*` tabs are BLANK because the outcome log does
  not carry them; `rrs_alignment` is blank because it is REACHABLE and not derived yet -
  `held_run_score.UNDERIVED_DIMENSIONS` keeps those two facts apart, and a blank is right where
  the question cannot be asked. `d1_setup_present` is fed from the scanner's own
  `master_avwap_tracker_scoring_snapshot.json`, never the 1.1 GB tracker, and its index
  **expires on the day roll**. **Every number on that table names its own basis**: the champion
  tier is a COLUMN (PROVEN / MUTED / active / blank) and the aggregator's verdict is headed
  **"Verdict (edge score)"** because it is computed from average R. **The My Decisions tabs
  carry the headline too** through `apply_held_and_ran`; those rows name no side, so
  `held_run_score.ALL_DIRECTIONS` gives them a pooled cell accumulated FROM THE EPISODES - never an
  average of the long cell and the short one. **Held is MEASURED held** (packet Q1, 2026-09-04:
  979 of 8,161 recent episodes read held with the question never answered): an episode is
  `measured_held` / `measured_broken` / `pending` / `unmeasured`, `hold_rate` is held / MEASURED,
  the unmeasured are COUNTED and SHOWN (the tracker's Measured column, `coverage`) and never
  assumed, and a stop first seen past the window with no earlier row bracketing it is
  `break_time_unknown` - the log carries no first-break time, and adding one is a `legacy.py`
  ask. **The D1 dimension is the ALIGNED same-session setup**: the join keeps the snapshot's
  `side` (`aligned` / `opposed` / `none` / `unknown`), only ALIGNED carries the privilege, a
  missing snapshot is UNKNOWN never False, and the basis is retrospective because the snapshot
  carries no time of day. The window is `evidence_stats.lately_window`, gaps reported.
- **The AWAY digest ranks swing picks by the tracker's record, not by the bucket** (answer 8:
  *"the cream is not being sent"*). Order is the **Wilson lower bound** on the family's realized
  win rate from `master_avwap_tier_outcomes.csv` inside `lately_window()` at ONE DECLARED HORIZON
  (`evidence_stats.SWING_HORIZON_SESSIONS`, 5, re-exported as
  `autopilot_core.SWING_DIGEST_HORIZON_SESSIONS`), expected R as the tiebreak; an ungraded family
  sorts BELOW every graded one, never at zero. **The horizon is declared because that file is one
  row per (pick, horizon)**: pooling horizons inflates n with correlated looks and CHANGES THE
  ORDER. A `stale_horizon` row is dropped. The bucket is PRINTED and never ranked on; the near cap
  is applied **after** ranking. The read is the caller's, so `render_away_report` stays pure.
  AWAY is still the only routine pusher.
- **The Research tab is not a trader surface** (answer 7). Nothing the trader must see may live
  only there; "it is on the Research tab" is not an answer to "where does the trader see this?"
- **"Lately" is ONE number and it is counted in trading sessions.** `evidence_stats.LATELY_SESSIONS`
  (20); `lately_window()` walks the exchange calendar (twenty calendar days is fourteen sessions
  in a normal month and twelve across a holiday week). `review_learning.DEFAULT_WINDOW_SESSIONS`
  IS `LATELY_SESSIONS` and the blind-spot and leak callouts are cut on it; Weekend Prep's week is
  `evidence_stats.WEEK_SESSIONS` (5). The state key, the report header, the CLI flag and every
  renderer say **sessions**.

## N3 - the research narration is bounded, and says what it left out (2026-09-05)

**Rule:** `narration_view` sends a BOUNDED view; the cells that fit are chosen by evidence
count, never by result, and every surface says `narrated K of N`.

**What broke.** The `setup_research` ledger read `narration absent` four nights running:
09-02 273,622 chars / 47 cells (the server sheared it at 32k), 09-03 82,192 vs the 78,119
budget / 69 cells, 09-04 143,636 / 128, 09-05 **658,292 / 619** - gate #59's lake recompute had
landed 141,299 recipe outcomes overnight (from 23,802). R3 (2026-09-02) had already cut the view
down from the pack and hoisted the repeated prose; what remained is that the view carried EVERY
eligible cell and the grid grows with the lake. No budget a 64k-context model can read holds
658k chars, so the refusal's own advice - "raise the budget or num_ctx" - could never be taken.

**What was built** (`scripts/ai_jobs/setup_research.py`: `_bounded_narration_view`,
`_policy_cell_order_key`, `_after_like_order_key`; BD-101). The head of the view (everything
that is not a cell: 11,084 chars live) is encoded first, then eligible policy cells are added in
order until the next would cross the budget, then the after-like ELIGIBLE cells (P10 C3,
unchanged) under the same rule. The order is `stats.n` descending, then `recipe_id`, `family`,
`side` - `stats.n` is the outcome-row count the eligibility floor gates on; `n_episodes` sits
beside it and was equal on all 619 live cells; the after-like grid is FLAT so it reads its own
top-level `n_episodes`. **The key knows how MUCH evidence a cell rests on and never how it
turned out**: gate #43 is a refusal, and a narration ranked by result would hand the model the
flattering half of a frozen grid. The lead's review perturbed every non-`n` statistic on the
live pack (26 fields, bootstrap bounds and trimmed means included) and the ORDER was identical;
only K moved, because the perturbed floats were longer. Coverage is stated as `narrated`
{K, of, selected_by, after-like K, of} in the view and the `.narration.json`, as one line under
`## Narration` in the pack markdown (computed BEFORE the file is written, so one pack and one
`.md` per date still holds), and as `narrated K of N eligible cell(s)` on the ledger reason.
The refusal narrows to "the head plus the FIRST cell does not fit" and names the head's size,
so a future `narration absent` says whether the head or the window is the problem. The evidence
hash is still over the bounded view - what was sent. Live dry run on a copy of the 09-04 pack:
658,292 -> 77,791 chars, **64 of 619 narrated**, the first five all `m5close_atr*` cells at
n=621.

**Live gate #67** replaces gate #40's narration clause: the next overnight `setup_research`
row reads `narrated K of N eligible cell(s)` with a `.narration.json` beside ONE pack for the
date, and the pack markdown carries the coverage line.

## ST5 - a window called sessions that counted days, and a bearish trader with a bullish record (2026-09-06)

Four defects, one theme: the personal-evidence chain named things it was not measuring.

**The window said SESSIONS and the arithmetic said DAYS.** `preference_trade_outcomes` had
`TRADE_WINDOW_DAYS = 10` and `window_end = said_on + timedelta(days=TRADE_WINDOW_DAYS)`, under
a docstring that read *"a trade opened on or within TRADE_WINDOW_DAYS SESSIONS after the
statement"*. The two disagree by more than a rounding: `market_calendar.trading_days_between`
puts ten sessions after Friday 2026-09-04 at **2026-09-21**, because 2026-09-07 is Labor Day,
while ten calendar days is 2026-09-14. Five sessions on the floor, and with them every trade
the trader took in the second week after saying something. `TRADE_WINDOW_SESSIONS` and
`statement_window_end` replace it; the old name stays one release as an alias because nothing
in `scripts/` imported it (grep, 2026-09-06). The FALLBACK matters: outside the calendar's
validated 2000-2032 range `market_calendar` raises rather than extrapolate, and the fallback
is the OLD, strictly NARROWER calendar-day arithmetic. A window that GREW on a refusal would
manufacture a link the trader never made; a narrower one only loses a link, which is the
direction uncertainty is allowed to fail in.

**One row per statement is right; one P&L per statement is not.** The live export on
2026-09-06 held 538 rows, **13 with `traded=yes`, over 10 distinct `trade_id`s**. Every row
is worth keeping - a statement with no trade is the skip, which is the most interesting row
in the file - but a summary summing `journal_net_pnl` across statements counted three trades'
money twice. `trade_level_summary` sums once per `trade_id` and publishes both denominators
plus `duplicate_statement_rows`, so the gap between the two grains is a number a reader can
see rather than a discrepancy they have to find.

**`trades.direction` is ownership and was being read as a market view.** The live journal:
**53 of the 89 option trades are SHORT and 39 of those 53 were winners.** Read "short =
bearish" and the desk describes a bearish trader with a bullish record. They are sold puts,
and a sold put is bullish-to-neutral. `scripts/journal_exposure.py` splits the question four
ways - instrument, ownership direction, market bias, structure - and answers each only as far
as the store allows. **A LONG option is never a bullish setup**: LONG CALL `bullish`, LONG PUT
`bearish`, SHORT PUT `bullish_or_neutral`, SHORT CALL `bearish_or_neutral`, stock follows
`direction`, and `UNKNOWN` (86 rows), `BAG` (1) and `CASH` (1) stay `unknown` and uncertain.

**A `trade_legs` row is a FILL, not a contract leg.** Every closed option trade in the live
journal carries at least two of them and 39 carry exactly two, so "more than one option leg"
would have called almost every ordinary option trade a spread. `multi_leg` means more than one
distinct option CONTRACT among the legs. The contract itself is not a column: `trades` has no
right, strike or expiry, so it is read from the OCC `trades.symbol` (`AA260522P00062000`) and
from `raw_executions.raw_json["option"]`, which `journal_statement_import._execution_from_row`
writes. `JournalStore.list_trade_legs` gained exactly ONE column, `e.raw_json`, to make the
second source reachable.

**`partial_of_spread` could not be derived and is therefore not claimed.** Two legs of a
spread arrive as two separate `trades` rows keyed by their own OCC symbols; nothing links
them, and this packet may not mint an identifier (plan.md P5.3/P5.4 own the canonical one).
What is observable is a second option trade on the same underlying and expiry opened in the
same session on a different contract - which is equally what two independent ideas on one name
look like. The label is `partial_of_spread_candidate`, it puts the row in the UNCERTAIN
population so its P&L never lands in a clean total, and it asserts nothing further.

**Populations are by STATUS; uncertainty is a LABEL across them - and the first cut had that
backwards.** `journal_analytics.personal_evidence_summary` partitions every trade into
`complete` (CLOSED) / `partly_closed` (CLOSED_PARTIAL) / `open_exposure` (everything else),
and carries the uncertainty question beside each as `n_uncertain` plus a cross-cutting
`uncertain` block. The first cut checked `exposure.is_uncertain` FIRST, which made uncertainty
a fourth bucket that ATE the other three. Reproduced on a copy of the live journal 2026-09-06:
**`uncertain` came out n=120, holding 84 CLOSED trades, ALL 7 CLOSED_PARTIAL and 29 of the 32
OPEN ones.** So `partly_closed` read **n=0** while seven exist; `open_exposure` read n=3 with
a notional of 7,726 against a real **61,662**; and ONE pooled P&L figure summed realized
results together with open positions' unrealized marks and **counted those marks as
WINNERS** - which is the exact defect this whole summary exists to prevent, one level down.
The correct numbers are 165 / 7 / 32 summing to 204, with `uncertain` a cross-cutting 120 split
84 / 7 / 29 that pools no money and no winners at all: its members span three statuses, and one
figure over a closed result and an open mark is the thing that went wrong. **An open position's
`net_pnl` AND `winners` are both `None`** - none, not zero - and its size travels as
`notional`. An EMPTY bucket reports `None` too: a net of 0.00 says "measured, and it came to
nothing", where a blank says "nothing here". The tester's fixture used STK for its
partly-closed and open rows, which is why it passed on a broken partition; the regression tests
use OPT and BAG there.

**No personal setup is called best without confirmed tags at the floor, and one tag is not
zero tags.** Live: **1 confirmed tag, 26 provisional, 145 needs_review, `planned_risk`
non-null on 0 of 204.** The first cut counted confirmed over CLOSED and provisional over ALL
rows - and the one confirmed tag sits on a **CLOSED_PARTIAL** trade (EAT, 2026-08-21), so it
fell out of the numerator while its 26 provisional siblings stayed in and the headline said
"No confirmed setup tags" about a journal that holds one. That is a false statement about the
trader's own work, not a conservative one. Both lanes now share ONE denominator - closed OR
partly closed, 172 - and the headline names the count it has: `1 confirmed setup tag - under
the n=30 floor (26 provisional awaiting review) - no personal setup can be called best.` "No
confirmed setup tags" is reserved for a true zero. Below `evidence_stats.MIN_REPORTABLE_N`
(30) `best_setup` stays `None`; above it the winner ranks on `swing_headline`'s Wilson lower
bound, the same rule every other trader-facing swing surface uses. The coverage line -
`Confirmed tags: 1 of 172 closed or partly closed trades. Provisional awaiting review: 26.
Planned risk recorded: 0 of 172.` - reaches the Journal's Analytics tab and Weekend Prep
through that one helper.

**The whole-journal pass is linear, because it runs on the Qt thread.** The Journal's Analytics
tab calls `personal_evidence_summary` through `build_analytics_summary`, so `classify_all` is
on the paint path. Comparing every option trade against every other one and re-parsing the
other's legs each time measured **130 ms at 1,020 trades**. Each trade's contracts are now
parsed once and the sibling question is answered from a `(underlying, expiry, session)` index:
**5.9 ms at 1,020**, 1.1 ms on the real 204. Same answers.

**A missing plan is a worklist, never a calculation.** `journal_r` is blank on all 538 report
rows because `planned_risk` is null on all 204 trades. An R worked backwards from what the
trade did is a statement about the outcome wearing the plan's clothes, so nothing in this
chain fills it: Weekend Prep lists the closed trades with no plan, newest first and capped at
the newest fifty with `showing 50 of 165` printed (the packet's ten-row floor was a MINIMUM
height, not a licence to build 165 `QTableWidgetItem`s on the Qt thread), and a row
only REFERS the trade to the Journal's Trades tab where `JournalStore.save_risk_fields` sits
behind the trader's own hand. Two tests spy that method into a raise and assert every reader
leaves it uncalled. `journal_feed.suggest_planned_risk`, the only prefill in the chain, was
checked in passing and is clean: it reads an ARMED ALERT's entry and stop - the trader's own
plan at decision time - and returns `None` on anything but a unique match.

**The tag backlog is wider than the week.** Weekend Prep's review list was scoped to the
current week while 26 provisional tags waited, most of them older - which is why gate #36
("confirm or edit at least ten of the 24") could not be worked from the screen built for it.
The provisional half is now the whole backlog; `needs_review` stays week-scoped, because those
145 rows carry no proposal and would bury the 26 that do. The week's rows sort first and a
`Week` column names which population each row came from.

**One deviation from the packet, and the reason.** The coverage line was asked for on Weekend
Prep's verdict card. That card is five to eight lines by the trader's own request (V2 item 2b)
and two tests pin it; a ninth line failed
`tests/test_v2_weekend_verdict_and_refresh.py::test_one_unreadable_store_still_leaves_a_card`
on the first full run. The sentence sits in its own label directly UNDER the card, filled from
the tag page's existing worker through `TagWeekPage.coverageChanged` - same screen, same
worker seam, no second read of the journal.

**Live gate #79.**
## ST3 - a stop filled at a level the bar never traded (2026-09-06)

### What was measured, on `main` @ `84ee24d6`

The review's fixture, reproduced by running the shipped `_evaluate_tracker_scenario_bar`
itself rather than reading the code:

| | entry | risk | hard stop | next bar | booked | R after the shipped costs |
|---|---|---|---|---|---|---|
| LONG, `literal_level_v1` | 100 | 5 | 95 | O80 / H85 / L79 / C82 | `HARD_STOP` @ **95** | **-1.014R** |
| LONG, the honest fill | 100 | 5 | 95 | same bar | the open, **80** | **-4.014R** |
| SHORT mirror | 100 | 5 | 105 | O120 / H121 / L115 / C118 | `HARD_STOP` @ **105** | 120 under v2 |

The bar traded between 79 and 85. Nothing traded at 95. Three more shapes behaved the same
way: a bar carrying no `open` at all still booked 95, a bar whose `open` was NaN still
booked 95, and a candle whose own prices contradict each other (O93 / H90 / L95 / C92,
`low` above both its `high` and its `open`) booked a fill off that `low`.

Separately, `calc_anchored_vwap_band_history` folds day D's own OHLC and volume into the
cumulative sums BEFORE it writes `history[D]`, so `history[D]` is a number that exists only
once D has closed - and `recompute_tracker_setup_record` handed exactly that dict to the
evaluator, whose target tests read the SAME day's `high` and `low`. The test that pins this
builds a day whose typical price sits on the running VWAP with three times the accumulated
volume, so folding it in leaves the VWAP where it was and halves the running deviation; the
day's own `UPPER_3` then sits INSIDE its own range while the previous day's sits above it.
`same_session_v1` books a target there. `prior_session_v2` does not.

### The rules this produced

- **A fill is a price that traded.** Under `gap_aware_v2` a bar that opened through the
  level fills at the OPEN (basis `gap_open`), a bar with no usable open fills at the level
  CLAMPED into `[low, high]` (`clamped_no_open`), and the price is always inside the bar -
  `resolve_fill` raises rather than returning one that is not.
- **The convention is SYMMETRIC.** A stop gap and a target gap are the same mechanic: a
  resting order whose price is already through at the open fills at the open. That makes
  the repair better for the trade as often as it is worse, and the measurement below says
  which dominates. An asymmetric "stops honest, targets at the level" convention is a
  different policy and would be a different version.
- **An invalid candle answers nothing and cancels nothing.** It books no fill
  (`invalid_bar`), no excursion is read off it, and the hold clock still advances: an
  invalid bar on the maximum-hold index DEFERS the `TIME_STOP` to the next valid bar with
  basis `deferred_invalid_bar` (lead decision, 2026-09-06).
- **Only the INTRABAR tests move under `prior_session_v2`** - the partial-target and
  final-target touches. The hard stop is `entry_price - risk x multiple`, fixed at entry
  and point-in-time clean already. The two-closes protective stop, the recorded
  `active_stop_level` and the maximum-hold force close are CLOSE-based and keep day D's
  levels, because at the close day D's levels are known.
- **A skip is counted, never read as "not hit".** A prior-session level that does not exist
  increments `intrabar_skip_reasons["no_prior_session_level"]` on the scenario.
- **A default run leaves no label.** The record carries `execution_convention` /
  `level_knowledge` only for a non-default run, and a default run POPS them, so a record
  replayed once under v2 cannot keep a name the desk did not use.
- **An invalid bar skips the WHOLE bar, not just the fill.** Under v2 no excursion
  (`max_favorable_r` / `max_adverse_r`) is read off an invalid candle and no unrealized
  mark is written from it, because both would be derived from the same contradictory
  prices. Under v1 all three are still taken - that difference is the point of naming the
  two conventions. The skip is counted as `skipped_bar_reasons["invalid_bar"]`.
- **The deferral label reaches the `TIME_STOP` and no other exit.** `time_stop_deferred`
  says "an unusable bar sat on the max-hold index"; an exit that fires ahead of the time
  stop on the same bar is its own decision and books `close` (or its own gap basis). The
  flag is cleared on EVERY path that closes the scenario, so a closed record never carries
  a stale `time_stop_deferred: True`.

### The comparison, on copies (ST3.3)

Seed 20260906 across the whole 11,372-record COPY of the SQLite mirror, daily bars from
the machine cache (633 symbols). **The denominator is `n_setups` 794** - 800 records were
OFFERED, 6 carry no tradeable scenario, 0 lacked bars, 0 failed to replay. `--limit` is
what was offered and is never the denominator; the first write-up quoted 800 and invited
exactly that confusion, so the CLI now prints
`population: n_setups 794 compared (offered 800, untradeable skipped 6, no cached bars
skipped 0, replay failed 0)` and the JSON carries a `population_note` saying the same.
The three JSON/CSV pairs live at
`%LOCALAPPDATA%\TradingBotV3\diagnostics\st3_execution_compare\`.

| run | changed | expectancy (raw) | win rate (Wilson lower) | R < -2 | groups moved rank |
|---|---|---|---|---|---|
| both repairs | 472 of 794 | -0.0981 -> -0.1192 | 0.576 -> 0.596 (0.541 -> 0.561) | 47 -> 48 | 43 of 50, max 14 |
| `gap_aware_v2` only | 89 of 794 | -0.0981 -> -0.0811 | 0.576 -> 0.597 (0.541 -> 0.562) | 47 -> 47 | 33 of 50, max 9 |
| `prior_session_v2` only | 458 of 794 | -0.0981 -> -0.1491 | 0.576 -> 0.548 (0.541 -> 0.513) | 47 -> 48 | 40 of 50, max 17 |

Three things the numbers say that the code alone does not. **The clip hides the tail**: the
first draft reported `min R -4.0 -> -4.0`, which is `TRACKER_SCORING_R_CLIP` (4.0) inside
`_summarize_tracker_setup_outcome`, not a measurement - a -6R gap fill and a -4R one are the
same number after clipping, so the artifact carries the CLIPPED R (what scoring reads) and
the RAW R (where the tail lives) side by side and never blends them. **The execution
repair mostly HELPS**: of the 89 setups it moved, 84 got better and 5 got worse, because
166 setups touched a `gap_open` and most of those are targets opening through their price.
The prior-session level knowledge is what costs expectancy (413 of its 458 moved setups are
worse). `min R` is -65.6011 under every policy - the sample's worst trade was already filled
inside its own bar.

And **the `invalid_bar` counter earned itself on the first run**: 380 scenario-bars over 26
setups / 22 symbols, and each of those 22 cached daily-bar files holds exactly ONE invalid
candle, all of them dated **2026-09-04**, every one with `low > open` or `high < open` -
AEE `O=105.81 H=106.96 L=106.11`, TWLO `O=239.52 H=239.29 L=231.21`, GPGI `O=14.195
H=13.925`. That is the signature of a FORMING bar written into
`%LOCALAPPDATA%\TradingBotV3\machine_cache\daily_bars` mid-session, where the "low" is the
low since the snapshot rather than the day's. Under `literal_level_v1` - what the desk runs
today - those bars are still read for fills, excursions and marks. This is a read of a
machine-local cache and NOT a claim about the tracker or the durable store; it is outside
ST3's scope and is recorded because an uncounted skip would have hidden it.
`no_prior_session_level`, by contrast, fired 4 bar-tests over 1 setup - real, and rare.

**Live gate #77** is the artifact plus a negative: the desk's next persisted tracker write
must carry no `execution_convention` and no `level_knowledge` key on any record, and no
event dict a `fill_basis`. The trader's decision on whether the repaired pair becomes the
scoring convention is asked separately; nothing on this branch takes it.
## ST1 - the family win was a favorable move at a scan-row offset (2026-09-06)

Trader, 2026-09-06: *"Name and version favorable price-direction observations separately
from simulated trade outcomes. Percent moves must not become realized R or stop-rule win
rates through wording."*

### What was measured, on `main` @ `84ee24d6`

`master_avwap_tier_outcomes.csv`, 19,558 rows, written 2026-09-04 13:07.

The `win` column is one line of `legacy.build_scan_factor_observation_rows`:
`"win": side_return_pct > 0`, where `side_return_pct` is the close-to-close move between
the scan row at `idx` and the scan row at `idx + horizon` - **that symbol's own scan
rows**, not exchange sessions. A long that broke its D1 support intraday and closed
higher on the target row is a `win` there; the stop-at-a-level, two-closes rule that
`swing_headline.headline_from_tracker_rows`' docstring claimed to be reading lives on
the tracker JSON scenarios (`_summarize_tracker_setup_outcome`), a different export at a
different grain. Four surfaces printed the first under the name of the second: the setup
docs sentence, the AWAY digest ranking, the Master AVWAP setups table's `Family Win %`
column and the Setup Tracker's family rows.

**The clock.** Of the 2,989 horizon-5 rows over the last 20 SCAN DATES, `sessions_spanned`
was 5 on 1,128, 6 on 699, 7 on 626, 8 on 409, 9 on 34, 10 on 20 and 11-18 on 73. The
`stale_horizon` flag fires at `spanned > 2x declared`, so it caught 2.4% of that: a
"5-session" cell is mostly 5 to 8 sessions, and the flag is not the thing that would tell
you. Over the desk's own "lately" window instead (20 exchange sessions,
2026-08-07..2026-09-03) the file has 2,642 horizon-5 rows, of which 55 are flagged stale.

**Three readings of one file.** `setup_docs._all_family_outcomes` and
`autopilot_core.swing_family_records` each wrote out the same three rules (one horizon,
drop explicit `stale_horizon`, bound to the lately window);
`legacy.build_bot_tier_performance_rows` read the same rows with no stale filter. One
file, one question, two answers - 5,005 flagged rows across the file.

**The tiers.** `tier_source` over the whole file is 19,217 `derived_from_bucket` and 341
`assigned`, and all 341 are horizon 1 from 2026-09-02 and 2026-09-03 (the first scans
after `assigned_tier` reached the feature history). So **every horizon-5 S/A cell the
desk shows today is built entirely from labels reconstructed from the priority bucket**,
and nothing said so.

### The rules this produced

- **A percent move is labelled a percent move.** `outcome_kind` is appended to the
  observation and tier-outcome headers with the v1 value
  `favorable_direction_scanrow_v1`; `swing_evidence.outcome_kind_of` reads a missing or
  empty cell as that, because every historical row was written before the column existed.
  `win` keeps its name and its value - identity and history are preserved - and the
  sibling column declares what it always meant.
- **The words follow the outcome kind, never a local string on a surface.**
  `Headline.outcome_kind` is `favorable_direction` from `headline_from_tracker_rows` and
  `trade_r` from `headline_from_outcomes`; `format_win_rate` renders `62% favorable
  (>=52%, n=90)` for the first and is unchanged for the second; the setups table's header
  is `Family favorable %` with the column key `family_win_rate` untouched, because the
  panel's widths, squeeze order and sort handler are pinned to the key.
- **A REPEAT and a COLLAPSE are different facts and are counted under different names.**
  `_scan_factor_row_id` is `symbol:scan_date:run_id`, so two scans of one symbol on one
  day are two SCAN ROWS, not one recorded twice. The first build keyed de-duplication on
  `(symbol, scan_date)` the way the v1 frame prep does, and on the live history (146,367
  scan rows) that reported **475,492 duplicates against 109,584 rows** when the truly
  repeated `scan_row_id`s number **75** (300 at four horizons): the desk ran 15 scans on
  2026-08-31 and the counter called 14 of each of them a duplicate. So
  `dropped_duplicates` is the true `(scan_row_id, horizon)` repeat count and nothing else.
- **One row per session, and it says how many scans stand behind it** (the trader's lead,
  2026-09-06, on a 127.5 MB-per-scan export: *"the v2 MEASUREMENT for one (symbol, side,
  scan_date, horizon) is the same number for every scan run that day"*). It is the entry
  session's close against the target session's close; neither moves because the desk
  looked again at 11:15. The build keeps ONE row per `(symbol, side, scan_date, horizon)`,
  the session's LAST scan row - the same choice `_prepare_scan_factor_history_frame` makes
  for v1, off the same sort, which is what makes the two files join **1:1** on
  `observation_id` - and `collapsed_same_session` carries the fold on the row AND as a
  builder total in SCAN ROWS. The log line names both counts; reporting them as one is how
  fourteen honest re-scans became "475,492 duplicates".
- **The v2 build is a ROLLING WINDOW and the file is not an archive.** It covers scan
  dates within `BUILD_WINDOW_SESSIONS` (30 exchange sessions, 1.5x the widest window any
  surface reads) of `last_completed_session`, and what falls outside is COUNTED in
  `excluded['outside_build_window']`. Measured through the export path on a copy of the
  live history at the shipped settings: **91,116 rows / 5.3 s / 25.6 MB** (91,880 scan
  rows folded, 300 true duplicates), against 110,308 / 5.7 s / 30.9 MB collapsed but
  unbounded and 458,336 / 13.4 s / 127.5 MB at the scan-row grain over 60 sessions. The
  collapse does most of the work and the window takes the last 17%; a settled row's target
  close does not move, which is what makes an unbounded rebuild pure rewrite.
- **v2 counts sessions, and it is a second file, never a replacement.**
  `master_avwap_lib/session_horizon_outcomes.py` walks the exchange calendar forward from
  the entry session, reads the bar ON the target session, and answers
  `no_bar_for_target_session` rather than reaching for the next bar. `sessions_spanned ==
  horizon_sessions` by construction. A target session after `last_completed_session` is
  `immature` and lands in `pending` - "the horizon has not arrived" and "the horizon
  arrived and could not be read" are different facts, and neither is a loss. A duplicated
  `(scan_row_id, horizon)` makes ONE row and is COUNTED, because
  `_prepare_scan_factor_history_frame` de-duplicates silently and a row nobody can
  reconcile is a row nobody can check. `observation_id` is computed the v1 way, so the
  two files join 1:1.
- **The export never fetches, and the shadow never costs the champion.** `closes_for` is
  supplied by the caller from the frames the scan already walked
  (`closes_from_daily_frames`); a symbol with no frame yields unmeasured rows with a
  reason. The whole v2 write is wrapped so a failure logs and returns - the v1 CSVs and
  the tracker save are never at risk for a file with no production reader.
- **One eligible-row reader, and it reconciles.** `swing_evidence.SwingOutcomePolicy`
  declares the six things the trader listed and `read_eligible_rows` applies them;
  eligible + pending + sum(excluded) == source rows, and every exclusion is named
  (`wrong_horizon`, `outside_window`, `stale_horizon`, `unreadable` - a present-and-empty
  horizon is not a zero -, `duplicate`, `unmeasured:<reason>`). `POLICY_SESSION_V2` reads
  the v2 file and **has no production caller**: it is the seam a later decision flips.
- **The tier performance export shares the RULE, not the policy - and its BASELINE is
  filtered like for like.** Its cells span every horizon at once over a 365-day lookback,
  so a policy's horizon and window clauses do not describe it; what must not differ is
  what an unmeasurable row means. `read_eligible_rows` and
  `build_bot_tier_performance_rows` therefore call the SAME function,
  `swing_evidence.is_stale_horizon`, and the export applies it to the observation rows it
  builds its baseline from as well. An edge is a cell minus its baseline; a baseline built
  on different rules makes that subtraction a comparison of two populations.
- **A coverage line is compared to ANOTHER READER, never to a number written down.** The
  window rolls on the exchange calendar, so the same file answers `2587 / 0 / 16971` at
  `end=2026-09-03` and `2462 / 0 / 17096` three days later - both correct. What a gate can
  check is that two readers of one file, asked in the same minute, say the same thing.
- **The window is asked of the row's own clock.** v1 has only its scan date; a v2 row
  knows the session it was MEASURED on, so `POLICY_SESSION_V2` windows on
  `target_session`. Maturity is checked BEFORE the window, so a pick whose horizon has
  not arrived is pending rather than "too recent to count".
- **A reconstructed tier never validates a shipped one.** Every tier-performance row
  carries `n_assigned_tier` / `n_derived_tier` from `swing_evidence.tier_split`, and
  `assigned_only=True` restricts a cell to the tier that shipped. A `tier_source` cell
  that is present and empty is UNKNOWN, never assigned.
- **v1's drift columns keep the business-day basis, deliberately.** `horizon_drift` gained
  `calendar=` and says "exchange sessions" when it is given one - Friday 2026-09-04 to
  Tuesday 2026-09-08 is ONE session and `numpy.busday_count` answers 2 - but the v1 export
  does not pass it. Doing so would rewrite `sessions_spanned` and `stale_horizon` on
  19,558 rows written on the other basis, which is a restatement of the file rather than a
  description of it, and the trader's 2026-09-06 prompt excludes exactly that.

**Live gate #75** is owed at the next persisted tracker write: the v2 file beside the tier
outcomes with `sessions_spanned == horizon_sessions` on every measured row, a reason on
every other, and the three readers printing the same `eligible / pending / excluded` line.


---

## ST6 - one snapshot, four surfaces, and a declared margin (2026-09-06)

### What was true before

Four surfaces on this desk answered *"what is working"* and nothing tied their answers to
one reading:

* the Setup Tracker's **BEST PERFORMING RIGHT NOW** banner, which ST2 had just put on
  `working_lately.select_leader`;
* the Summary card three lines above it, which ST2's fix round put on the same function
  after it was found crowning `max(avg_closed_r)` across both namespaces;
* Weekend Prep's verdict card;
* the AWAY Recap.

Each read the files itself, at its own moment, with its own `previous`. Two of them could
name different families on one afternoon and there was no way to tell which was older. The
**Trading Desk itself had nothing** - decision 0016 says *"what is working lately"* belongs
there and never only in Research, and the trader's own screen was the one surface with no
answer at all. The priority switch had been written down in `CLAUDE.md` as *"NOT BUILT
YET"* since V4 was scoped, with the identical-visible-rows test owed WITH it.

A fifth thing was true and nobody had noticed: **`held_run_score.Segment.summary` called
`evidence_stats.summarize` with the VALUES ALONE.** No `symbols=`, no `sessions=`. So
`concentration.by_symbol`, `concentration.by_session` and the session-block `bootstrap`
came back UNMEASURED for every held x ran cell the desk has ever drawn. The day-trade
headline - decision 0016's own answer for every day-trade surface - had no way to say it
was one name six times. The episodes have carried `symbol` and `trade_date` since V1;
nothing had to be measured again, only handed over.

### The rules this produced

**The snapshot is PURE, and its identity is the EVIDENCE.** `working_lately.build_snapshot`
opens no file, starts no thread and asks no store a question; the service reads on a worker
and hands the rows in. `snapshot_id` is a sha1 over the SORTED cell tuples, the declared
policy lines and `as_of` - and over nothing else. Not `built_at`. Not a source's mtime. Not
the verdicts. The reason is the events file: a snapshot whose identity moved on every
half-hourly tick would make `leader_change_events.jsonl` a log of the timer, and one that
moved because a file was re-saved unchanged would announce news that never happened.

**Three kinds, three verdicts, and `pool_cells` is a refusal.** A swing trade-R rate, a
swing favorable-direction rate (ST1: the tier outcomes' `win` is the sign of a percent move
at a scan-row offset) and a day-trade `held_run_score` are three questions with three
outcome definitions, three horizons and three units. `pool_cells` is the one helper that
would combine cells and it RAISES across kind, side or outcome kind, naming the axis. There
is deliberately no formula behind it - the refusal IS the mechanism.

**Dependence is answered by refusing; multiple testing is answered by printing.** The
trader's condition was explicit: *"Any new confidence calculation must account for shared
sessions and overlapping holdings and must be validated; ordinary Wilson bounds alone do
not solve dependence or multiple testing."* A validated one was not affordable inside this
packet, so none was written. Instead a cell whose top symbol or top session supplies MORE
than `CONCENTRATION_LIMIT` (0.5) of its own sample is **not eligible to lead**, reason
`concentrated` - and the exposure from having looked at K cells is PRINTED on every surface
as `observational leader among K cells`. Nothing is called proven. The limit is `EXCEEDS`
and not `reaches` on purpose: a cell split evenly over two sessions sits at exactly 0.5 and
is the smallest honest spread the desk sees, so refusing it would refuse the ordinary case.

**Both numbers were declared before any forward look.** `LEADER_PERSISTENCE_SNAPSHOTS` = 2
and `CONCENTRATION_LIMIT` = 0.5, 2026-09-06, per the trader's *"Choose any margin/persistence
rule before inspecting its forward evaluation."* Two is the smallest number that is not one:
it costs a real leader a single session and it stops a one-session wobble - a correction
landing, a stale export, one heavy name reporting - being announced as a change of regime.
Until it is met the verdict is `no_clear_leader` with `awaiting persistence (1 of 2)`, which
is a true sentence about the evidence rather than a hedge.

**Cause precedence is REFUSAL-FIRST.** The packet listed `window_rollover` first ("when
`as_of` moved"), but `as_of` moves on nearly every build, which would have made
`corrected_data` and `lost_coverage` unreachable - the two causes actually worth reading. So
the order is `lost_coverage`, `corrected_data`, `window_rollover`, `new_outcomes`. A
`lost_coverage` event carries the SAME name in both slots, because nothing new was learned:
the desk simply stopped being able to read it. And a session bucket that merely EMPTIED
while the source's total row count held steady is a window moving forward, not a
restatement - the recent-types export re-states one row per family under a new
`latest_measured_session` at every close slot, so every previous session's bucket empties on
a perfectly ordinary day.

**An absent source is `rows is None`, never `0`.** A zero is an answer; an unreadable file
is a question that was not asked, and the verdict says `source unavailable` rather than
producing a leader out of nothing.

**A share is a DECISION input, so it is not display-rounded.**
`evidence_stats._concentration` rounded `top_share` to four places, which made 2/6 read
0.3333. That share is now compared against a declared 0.5 and hashed into a snapshot id, and
a display rounding inside a decision is how a cell sitting on the limit lands on the wrong
side of it. Ten places: far below anything any surface prints, far above float noise, and
every renderer formats it itself.

**The switch reorders and never withholds, and the test is behavioural.**
`prioritise_working_lately` is read AT SORT TIME and never at write time - the backing lists,
every evidence write, the tier gate, the movers-only filter and the repetition fold are all
computed before any sort. The WAITING list is sorted where `_advance_review_queue` picks the
next chart, not where `_enqueue_review_alert` writes a row, because the backing list is the
record of what the day produced and it must not depend on a display preference. The test
drives all three surfaces through their real paths twice and asserts the same set of names,
the same repeat badges and the same hidden set both ways, with only the order different.

**The strip is mounted inside the alert bar, not beside it.** The M5 column is a saved,
draggable two-pane splitter (the alert list over the swing favorites strip). A third child
would have the trader's own saved sizes replayed onto a layout they never dragged, so the
strip goes in at layout index 0 of `M5AlertBar` and `m5_column.widget(0)` is still the bar.

**A cell keeps the case its own source spells.** The swing exports say `LONG`; the intraday
outcome log says `long`. Rewriting either inside the snapshot would make the cell disagree
with the file a reader opens next, so the case travels and every COMPARISON upper-cases
(`EvidenceCell.name`, `pool_cells`, `priority_rank`).

**Live gate #83** is owed at the first DESK session after merge: the strip with a verdict per
kind, `snapshot_latest.json` carrying the same `snapshot_id[:8]` the banner prints (and the
words `panel read` absent from it), the switch reordering the M5 list with the same row
count, at most one event per kind per session, and a restart adding none. Expect the first
session's swing verdict to read `awaiting persistence (1 of 2)`: the first build has no
predecessor, so a named leader on day one would mean the rule did not run.

## ST4 - the rescan that happened to close was the one that counted (2026-09-06)

### What was true, read on `main` @ `84ee24d6`

The Setup Tracker rescans a thesis every day it still looks like a setup, so one
`(symbol, side, anchor_date, setup_family)` leaves many rows behind and something has to pick
which one gets graded. Three different places made that pick by reading the OUTCOME.

1. **`_dedupe_recent_tracker_family_rows`** sorted `(0 if closed_setups > 0 else 1, scan_date)`.
   A closed record wins, and only then does the earliest date. So the 08-10 open entry a trader
   could actually have taken lost to the 08-15 rescan that had resolved by the time the tracker
   looked. The docstring called the second key "the trade you would actually have taken at first
   signal", which is true only when no row in the group has closed.
2. **`_representative_scenario`** fell to `matching[0]`, and scenario insertion order is
   `_build_tracker_scenarios`' `for stop in stop_candidates: for template in
   SETUP_EXIT_TEMPLATES`. `REPRESENTATIVE_EXIT_TEMPLATE_ID` was `""` - the comment above it said
   the choice was decided by dict order and that nothing had ever said which template the headline
   R was measured on. Reordering the `scenarios` list moves one setup's headline **from +2.00R to
   -1.00R** (`tests/test_st4_first_actionable.py::test_representative_exit_template_survives_a_scenario_dict_reorder`
   pins BOTH readings).
3. **`_summarize_tracker_setup_outcome`**: `representative_closed_r = rep_total_r if (rep_total_r
   is not None and rep_is_closed) else avg_closed_r`. When the representative is still OPEN, the
   headline becomes the mean of the OTHER closed scenarios - the alternate exit plans nobody
   chose. A setup whose `full_band2` representative sits at +0.40R open, beside a `full_band3`
   that closed +3.00R, **reports +3.00R for a trade still running**.

### The premise the packet asked to verify

**There is no exit-date field on a scenario.** `_apply_scenario_exit_event` appends one entry per
exit LEG to `events` (`trade_date`, `reason`, `price`, `shares`, `pnl`, ...), so the recorded exit
is the `trade_date` of the LAST entry, and an open scenario has that list PRESENT and EMPTY.
`_scenario_recorded_exit_date` gates that read on the CLOSED status, because a PARTIAL leaves a
dated event behind on a scenario still in the trade: read as an exit, it would let the challenger
open a second attempt while the first one was still running - the exact thing the re-entry rule
exists to prevent.

### The rules this produced

**The default did not move, and that is the point.** `closed_first_v1` is
`DEFAULT_SELECTION_POLICY` and every caller still gets it, byte-identical
(`tests/fixtures/st4_family_rows_golden.csv`, pinned from `main` before any of this code existed,
reproduced under the bare default call AND under an explicit `DEFAULT_SELECTION_POLICY`).

**The shipped rule has exactly ONE implementation.** `_v1_pick` in
`scripts/master_avwap_lib/selection_policy.py` is the old body moved verbatim; the legacy function
delegates. A challenger measured against a second copy of the champion measures the copy.

**An episode under `first_actionable_v2` is `(symbol, side, anchor_date, setup_family,
attempt_index)`.** Attempt 1 is the EARLIEST scan row - chosen on the date alone, before any
outcome exists. A later row opens attempt k+1 **only** when the previous attempt's representative
scenario closed strictly before it (`REENTRY_RULE_V2`): that is a declared entry rule, not "the
rescan that happened to close". A rescan while the attempt is live is one more OBSERVATION of the
same episode. An open attempt stays `pending` and grades nothing.

**The representative exit is DECLARED, and declaring it reads no outcome.**
`REPRESENTATIVE_EXIT_TEMPLATE_ID_V2` is `full_band2`, the first baseline entry of
`SETUP_EXIT_TEMPLATES` - the one dict order has always practically handed back. Naming it fixes
the answer against a reorder without preferring a better-performing plan.

**`excluded_reasons` has two grains and the token name says which.** A bare `reason=N` counts
records the population never admitted and is summed into `n_excluded`; an `_in_population` token
counts counted episodes a decision deliberately KEPT and is never summed. That is how the
2026-09-06 decisions are NAMED without being reopened: `untradeable` is (c),
`expired_unmeasured_in_population` is (b) - it stays in the champion's scoring population - and
(a), swept-measured M5 trades, is in another file and is named in the comparison's README.
`no_representative_in_population` is a builder addition on the same principle: the packet listed
`no_representative` as a drop, but no such record is dropped today and making one droppable would
have changed the default's numbers, so it is named and kept.

### The comparison, measured

Provenance, precisely: the live SQLite mirror (`master_avwap_setup_tracker.sqlite`, 1,191,460,864
bytes, `data_session` **2026-09-03**, 11,372 setups / 401 controls / 3,992 studies) was COPIED,
and a scratch script read the copy through `TrackerStore.load_records("setups",
scan_dates=<28-day window>)` and wrote a **windowed JSON extract of 5,696 setups**. The CLI ran on
that extract, not on the mirror. Both policies at the same `as_of_session` (2026-09-03) and the
same 28-day lookback. **32 cells, 27 changed, 26 rank moves.**

| | episodes | pending | wins-losses | unweighted win rate | Wilson lower | episode-weighted mean R |
|---|---|---|---|---|---|---|
| `closed_first_v1` | 2,249 | 498 | 1,078-673 | 61.6% | 0.593 | -0.150 |
| `first_actionable_v2` | 2,712 | 915 | 1,294-503 | 72.0% | 0.699 | +0.073 |

`n_excluded` is 22 under both and `fully_excluded_groups` is 0. Largest single cell: LONG /
favorite_setup / `avwape_to_1stdev`, 1,029 observations, 301 -> 409 episodes, 61.1% -> 73.4%,
mean R -0.269 -> +0.002.

**v2 bundles TWO policy questions, so the move is decomposed and neither half is answered here.**

1. *Selection, or pending-stays-pending?* Overwhelmingly the second. Hold v2's SELECTION fixed and
   grade it the old substituting way: 1,313-755, 63.5%, mean R **-0.138**. Honour
   `representative_status`: 1,294-503, 72.0%, mean R **+0.044**. On the family-weighted figure the
   whole move is -0.150 -> +0.073 and the substitution part is **+0.182 of +0.193, 94.2%**.
2. *Is the win-rate move the 463 re-entries?* Under the OLD aggregation it is nothing else:
   attempt >= 2 graded **72.5%** (235-89) against attempt 1 at **61.8%** (1,078-666), and v1's own
   rate on those same 2,249 theses is 61.6%. After the fix both rise and the gap survives: attempt
   1 **70.5%** (1,061-445), attempt >= 2 **80.1%** (233-58). Histogram: 2,249 first attempts, 393
   seconds, 61 thirds, 9 fourths. **A second attempt exists only because the first one CLOSED, so
   the attempt >= 2 population is survivorship by construction.** That is a question for the
   trader; naming it is this packet's job, answering it is not.

So the review's premise - that v1 flatters the rate - is not what the measurement shows. v1
under-counts EPISODES (a genuine second entry is folded into its predecessor) and it over-counts
GRADED ones (a pending representative was graded from the alternate plan's R). Those pull in
opposite directions on the headline.

### The two the reviewer caught

**The compact scoring projection IS the record.** `_build_scoring_projection` writes a projection
with no `scenarios` key and a `_scoring_outcome_summary`;
`master_avwap_tracker_scoring_snapshot.json` holds 11,372 of them, and that summary is the only
copy of the answer. The first ST4 build refused a cache lacking `representative_status` and
recomputed, which on a scenario-less record returned `tradeable_scenario_count == 0` and dropped
every setup: on a COPY of the live snapshot, `build_recent_tracker_setup_family_rows` gave **32
rows on base and 0 on the branch** and `build_tracker_setup_type_rows` **74 nonzero `score_delta`
on base and 0 on the branch**, which `runner.py`'s first D1 scan after a merge would have written
into live `recent_tracker_score_delta` / `setup_type_score_delta`. **A missing key is never a
reason to recompute.** A default read with no `as_of` takes the cache exactly as it did before
ST4; a challenger or replay read of a scenario-less record answers `representative_status:
"unknown_compact"` over the cached numbers, and under v2 such an episode is neither graded nor
pending but NAMED (`unknown_compact_in_population`).

**Pending has to stay pending in the AGGREGATE too, or the rule is decorative.** ST4.2 stopped the
per-setup substitution, but the family loop selected `closed_rows` on `closed_setups > 0` (any
tradeable scenario closed) and fell back to `avg_closed_r` when the representative's R was None -
the same substitution, one level down, grading **271 of 2,712** v2 episodes whose representative
was still running, 252 of them as losses. `_row_is_graded` is the single seam; under v1 it is the
identical `closed_setups` test and every number is unchanged (golden).

### A replay is blind to a compacted record

History compaction empties a scenario's `events` (`scenario["events"] = []`), and
`_scenario_recorded_exit_date` reads the last event, so a compacted CLOSED scenario cannot be
dated and an `as_of_session` build reads it as still running. **0 of the 141,324 scenarios in the
28-day window are compacted**, so the comparison above is unaffected; **99,562 of 206,341 across
all history are**, so an earlier cutoff or a longer lookback walks straight into them. The rule:
the row carries `representative_exit_undatable` and the family row names
`undatable_exit_in_population=N`, because "still running" and "we cannot see when it closed" are
different facts and only one of them is a trade still on. Related: a `(side, bucket, family)`
whose every record was excluded produces NO row, so the build-level `fully_excluded_groups` is
stamped on every row - the one number this accounting could otherwise lose with the group.

### A scoring snapshot is not a tracker, and the tool says so

The same compact-projection fact has a third edge, found on the re-review. Hand
`tracker_selection_compare` a copy of `master_avwap_tracker_scoring_snapshot.json` and the two
policies disagree for a reason that has nothing to do with either: v1 answers every setup out of
the cached summary - a cache written with no cutoff, so a replay grades trades it could not have
seen - while v2 cannot evaluate a scenario-less record and zeroes. Measured on the four-projection
fixture: `v1 episodes 4 pending 0 wins 3 losses 1` beside `v2 episodes 4 pending 0 wins 0 losses
0`. That report reads "v2 is broken" when the input was simply the wrong file. The CLI now counts
setups carrying `_scoring_outcome_summary` with no `scenarios`, prints one line naming the
snapshot, exits 2 and writes nothing. And `_row_is_unmeasurable` applies under **both** policies
whenever an `as_of_session` is given, so a v1 replay of a compact record grades nothing either -
the default read (v1, no `as_of`) is untouched, because `unknown_compact` is a status only the
bypass path can write.

**Live gate #78.** Nothing here promotes. The desk's next persisted tracker write must leave
`selection_policy = closed_first_v1` on every recent family row AND leave the scan's
`recent_tracker_score_delta` / `setup_type_score_delta` nonzero (32 / 74 on the 2026-09-05
snapshot is the reproducible check); the artifact stays under
`%LOCALAPPDATA%\TradingBotV3\diagnostics\st4_selection_compare\`; and the policy decision is the
trader's, as a separate change with its own golden fixtures.


### ST6 re-review - four things the first build got wrong (2026-09-06)

**A display preference that cannot be undone is not a display preference.** The
switch re-sorted the M5 bar's QListWidget and rebound the review queue's own
list, so a session that turned it on could never get today's arrival order back:
the bar returned early when disabled and the sorted list simply stayed. The rule
is now structural - the backing list is ARRIVAL-ordered and never touched, the
priority order is a VIEW computed at draw time, and the review queue is not
sorted at all (`_next_review_index` chooses an index). "Reorders and never
withholds" has to be reversible or it is a rewrite of the record.

**A cap over a sorted list deletes a different thing.** The `MAX_ROWS` trim took
the tail of whatever order was displayed, so with the cap at 3 and the oldest
arrival the highest-ranked cell, ON kept AAA/CCC/DDD and OFF kept BBB/CCC/DDD -
the switch deciding which alert stopped existing. The cap belongs to the arrival
list; the view is ordered afterwards.

**An interval belongs to the number it stands beside.** `held_run_score` is
P(held 30m) x the trimmed-mean MFE_R of the held ones - a product over two
denominators - and the bound printed next to it was
`session_block_bootstrap` over the MFEs alone. Live that read
`held x ran 1.21 (>= 2.070)`: a lower bound ABOVE its own statistic, and the cell
it ranked first (`LONG regime_pause_rs`) was not the cell with the best held x
ran (`LONG lrsi_cross_50`). `evidence_stats.session_block_statistic_bootstrap`
keeps the resampling in the one statistics contract and takes the FORMULA from
the caller; `Segment.score_bootstrap` recomputes the whole score on each draw.
**The day-trade kind ranks on the statistic** - it is the desk's declared
day-trade headline (decision 0016 answer 4) - and its margin is
`LEADER_MARGIN_HELD_RUN_R` (0.10, score units), declared 2026-09-06 before any
forward look. The 0.05 win-rate margin is a margin on a quantity bounded in
[0, 1] and never applies to an R scale.

**Freshness asks when a thing was MEASURED, not when it was entered.**
`swing_favorable` was dated by `scan_date` while the outcome is measured
`horizon_sessions` later, so the live file - newest `scan_date` 2026-08-28,
horizon 5, read against 2026-09-04 - was exactly one horizon behind and could
never be fresh. The cell's clock is `future_scan_date` (v1) or `target_session`
(v2). And a kind that is withheld now SAYS SO on the strip: it used to be printed
only when it named a leader, so silence read as though the desk had two questions
instead of three.

**Two smaller rules from the same round.** The observational caveat counts ONE
KIND's cells - the swing leader was never chosen against the day-trade cells and
`pool_cells` refuses to make them comparable. And a share is not display-rounded
and a concentration that was never taken says `concentration unmeasured` rather
than `top symbol unmeasured`, which reads like a measurement that came back
empty.
**Live gate #78.** Nothing in ST4 promotes. The artifact stays under
`%LOCALAPPDATA%\TradingBotV3\diagnostics\st4_selection_compare\`, and the scan's
`recent_tracker_score_delta` / `setup_type_score_delta` must stay nonzero (32 / 74 on the
2026-09-05 snapshot is the reproducible check). Its `selection_policy = closed_first_v1`
clause was written while the decision was still owed and is superseded by gate #84 below.

## ST7 - the three decisions became the defaults, and a record has to say which (2026-09-07)

The trader took two of them on 2026-09-06 ~21:15 PT - *"Yes a trade not yet completed should say
pending. A second entry after a first close is its own trade yes. 3. This one is up to your
discretion."* - and the lead took the third under that discretion. Decision record 0019 carries
the reasoning, the numbers and the rollback; this entry carries the three things the CODE does
that a reader would otherwise have to infer.

**Why the stamp became unconditional.** ST3 wrote `execution_convention` / `level_knowledge` onto
a record only for a non-default run and POPPED them otherwise, so a record replayed once under
`gap_aware_v2` and then replayed again by the desk could not keep a label the desk did not use.
That was exactly right while there had never been a flip: absent meant "the default", and the
default was one thing forever. After 2026-09-07 absent means "written by some earlier version
under some policy", which is not a fact anyone can act on. So `recompute_tracker_setup_record`
writes both stamps on every run, including the v1 one, and `selection_policy` is on every family
row and every `_scoring_outcome_summary`. The golden comparisons exclude the three stamp keys
from byte-identity and assert them separately by name, so no stamp escapes assertion in either
direction.

**Why there is now a log line at the tracker write.** There was none - not a success line, not a
count - so the packet's "state it in the log line the tracker write already emits" had nothing to
extend. One was added at the CALL SITE (before `save_setup_tracker_payload`, not inside it,
because the payload does not know which policies rebuilt it) reading
`Setup tracker policies: selection=<..> execution=<..> levels=<..>`, logged unconditionally: an
absent line and a line naming the v1 policies are different facts, and live gate #84 greps for
the token.

**The rule the flip needed that the packet did not name: an ABSENT `representative_status` is not
a pending trade.** ST4's cache rule says a DEFAULT read of a compact scoring projection takes
`_scoring_outcome_summary` verbatim, and it still does - `master_avwap_tracker_scoring_snapshot.json`
holds 11,372 records with no `scenarios` key at all, and for the live scoring path that summary is
the only copy of the answer. But v2's "pending stays pending" keys on `representative_status`, and
a projection written BEFORE ST4 does not carry that column at all. `_row_is_graded` therefore
reads the row's own `closed_setups` when the key is ABSENT, and `no_representative_in_population`
counts those rows so the gap is visible rather than silent.

**The size of the failure, measured rather than assumed** (reviewer, on a COPY of the 2026-09-06
snapshot with the bridge reverted): the 32 recent family rows still appear - this is NOT the first
ST4 build's total collapse - but their graded population goes **1,688 closed -> 0**, all 2,195
episodes reading pending, and the nonzero `recent_tracker_score_delta` count goes **14 -> 7**. The
`setup_type` deltas are **74 either way**, because `build_tracker_setup_type_rows` takes no policy
argument at all and never consults one. Half the champion's recent-family score input going blank
for one scan is a smaller fault than "everything zeroes" and is still the fault the bridge exists
to prevent; quoting the bigger number would have been quoting the wrong incident.

**The bridge is keyed on the key being ABSENT, never on the value being empty**, and the two are
different measurements. A summary ST4+ wrote for a record whose scenarios carry no representative
has the key with an EMPTY string - 48 such records on that snapshot - and it means "measured, and
there is no representative", which is unmeasurable, not closed. Those rows are NOT bridged; 13
episodes move pending -> unmeasured, so ST4's quoted 915 pending reads 902 + 13 on this branch.
The coerced string on the row destroys the distinction, so the row carries
`representative_status_known` (not exported; `TRACKER_RECENT_FAMILY_COLUMNS` is unchanged). A
status that is PRESENT and reads `pending` is still never graded - that is the decision. The next
persisted tracker write rebuilds every projection with the column filled, so this is a bridge, not
a permanent second rule.

**A record with NO representative scenario is UNMEASURED under the default, and that reaches the
STUDY families.** `_representative_stop_label_for_setup` answers the protective band (`LOWER_1`
long, `UPPER_1` short) unless the setup carries a first-band bounce signal, and
`_representative_scenario` returns None when no tradeable scenario carries that stop label. v1 did
not care - it graded on `closed_setups > 0` - so this never showed. The 1st-dev-breakout study's
scenarios are stopped at the VWAP, so under the default their rows are produced, counted, and
reported as `no_representative_in_population` with `closed_setups` 0 rather than graded. That is
ST4's rule arriving somewhere ST4 did not measure, it is shadow evidence either way, and no
champion pick is graded through it - but a reader of the study tables must not mistake the change
for the study going quiet. It is pinned by
`tests/test_first_dev_breakout.py::test_recent_family_rows_include_1stdev_breakout`, which now
asserts BOTH readings.

**The bypass moved with the default and the rule did not.** `unknown_compact` is what a read that
cannot be answered from the cache says instead of returning an empty summary. That is any
NON-default read - which is now `closed_first_v1` by name - or any `as_of_session` replay under
either policy. Naming `first_actionable_v2` explicitly is the default read and takes the cache.

**A test whose subject is one axis names the other.** Several ST3 cases isolate the execution
convention while booking a target off a level; under the new default level knowledge that target
is read from the PREVIOUS session, so with no `prior_session_levels` handed in they would have
stopped booking and passed for the wrong reason. Those cases now name `same_session_v1` on both
arms, or hand the same level in as the prior session's. The same applies to the Phase 0.10 B-2
band-variant parity fixture: it is read under the policies it was FROZEN on, and a separate test
re-runs its real subject - that a `VARIANT_*` scenario never enters a champion aggregate and never
displaces a champion scenario - under whatever the defaults currently are.

**Live gate #84.** The first persisted tracker write after merge logs the policies line, every
record carries the three stamps, `n_pending` is non-zero, no `pending` representative is graded,
the tables re-rank (expect the ST4 comparison's 2,249 -> 2,712 episodes and 61.6% -> 72.0%
favorable), and the first D1 scan after it still writes NONZERO score deltas.

## G5 - a window that labelled and never filtered, and a source line that printed `None` (2026-09-07)

### What was measured, on `claude/g5-research-results` @ `85d11227`

The Results page's window control (`Recent 20 sessions | All history | Custom...`) reached
`research_results._window_of`, which returned a NAME, two dates and a LABEL - and nothing else
read them. `_bot_sections` rendered every cell in the snapshot and `_mine_sections` bucketed
every closed trade in the journal, under whichever heading the buttons happened to be showing.
The reviewer reproduced it in one line: a trade closed in **2019** was counted, with its money,
under "Custom window 2026-09-01 to 2026-09-04". A heading is a claim, and this one was false in
every selection but "All history".

The same page's freshness line read, on every real snapshot on this machine:

    swing_trade_r <- an unnamed file @ None; swing_favorable <- an unnamed file @ None; ...

because `working_lately.build_snapshot` writes `path: ""` and `mtime: None` for each source it
computes itself (`_rows_by_session` fills in `rows` and `rows_by_session` instead), and the
first cut of `_bot_freshness` printed the path and the mtime and nothing else. The fixture the
tests were written against carried a path and an mtime, so nothing was red.

And a band card printed its BAND's size over the three rows it renders per section: 27 cells,
six printed lines, one label reading `27 shown`.

### The rules this produced

- **A window control either applies or is disabled; it never merely labels.** For **My trades**
  `research_results.in_window` filters CLOSED trades on `closed_at`, inclusive at both ends,
  and the count it turned away is reported (`stats["n_outside_window"]`) - a filter that drops
  rows silently is the same defect one step later. A trade whose `closed_at` the journal never
  carried is counted OUT of a bounded window: "not known to be inside" is not "inside".
- **For Bot setups the SNAPSHOT owns the window.** Every `EvidenceCell` carries the
  `window_sessions` its own aggregator walked, ending at the snapshot's `as_of`, so the three
  buttons are DISABLED with `WINDOW_ON_BOT_TOOLTIP` and the page prints
  `ResultsView.window_sentence` ("The snapshot owns this window: 20 sessions ending
  2026-09-04") in place of a date range its numbers never saw. `window_applies` is the flag a
  reader checks; it is False for bot and True for mine.
- **The three control groups react on `idToggled` filtered on `checked`, not `idClicked`.**
  `QAbstractButton::click()` returns early on a disabled button, and one of the packet's own
  tests drives the window control while the page is on Bot. Filtering `toggled` keeps ONE
  reaction per change (an exclusive group un-checks the old button as it checks the new one)
  and leaves a disabled button's STATE drivable, which is what a test does and a trader cannot.
- **A provenance line states what the snapshot recorded and never a Python literal.**
  `_source_text` prints the source's `rows`, adds a path or an mtime only when one is present,
  and a snapshot with no source at all prints `NO_SOURCES`. The lesson under it is older than
  this page: a fixture richer than the file it stands for cannot fail on the file's shape, so
  the test for a provenance line is written against the WRITER's own output.
- **A card counts the lines it rendered.** `N of M shown` - N the printed lines
  (`CARD_LINES`), M the band's own total - and an empty section prints no heading.
- **Every running-text label on a page is capped at G3's reading measure.**
  `READER_MEASURE_CHARS` is 100 characters of the label's own font, floored and ceilinged in
  design pixels, left-aligned, wrapped. Measured before the fix: the section label was ONE
  1,922-character line set across 3,456 px. The constant is asserted equal to
  `market_journal_panel`'s, so the two readers cannot drift apart.

## G7 - the pages the desk built for nobody, and the bench that taxed its own wait (2026-09-07)

### What was measured

The G0 baseline (2026-09-06, offscreen, `--repeat 3`) had three ops over the 250 ms sync
mark at 3456 x 2160: `research.construct` at 5.1 s, `setup_tracker.refresh` at 1.3 s and
`market_journal.construct` at 299 ms. The G7 recon found the shape behind all three.

- **No page was built lazily.** `app.py` constructed every left-nav page eagerly before
  the `QStackedWidget`. The ONE deferred first load in the whole app was the Market
  Journal's `showEvent` gate, whose own comment states the rule the rest of the desk was
  not following: *"the desk builds every left-nav panel at startup and most are never
  opened."*
- **`ResearchPanel.__init__` built nine children and five of them READ inside their own
  constructors** - the Day-trade Tracker (a CSV parse, a JSON read, every dimension model
  rebuilt and fitted), the Setup Tracker (`refresh()`, the 1.3 s op), Market Prep (two
  local reads plus the human-focus picks), Price Alerts (`load_price_alerts()`), and the
  Setup Playbook, whose `setCurrentRow(0)` rendered an overview whose banner did two
  UNCACHED CSV reads on the Qt thread.
- **`SetupTrackerPanel.refresh()` was entirely on the Qt thread** except the attribute
  leaderboard: twelve `_load_csv_rows_cached` calls, `load_human_focus_performance_rows`,
  thirteen `model.set_rows` and thirteen `fit_columns()`. The mtime cache stat'ed every
  file on every call and skipped only the PARSE, so the thirteen model resets and thirteen
  column fits ran even when nothing had changed - on every spinbox step.
- **The bench was taxing its own wait.** `desk_bench.settle` called
  `_widget_workers_running` on every poll, which walked `findChildren(QThread)` AND
  `findChildren(QWidget)` plus `vars()` over the whole page. The G0 reviewer measured
  ~1.5 ms per poll on Research against a 120 ms `QUIET_MS` window: the bench held the GIL
  for a large share of the time it was reporting as the page's.

### The rules this produced

- **A Research child reads on its FIRST SHOW, never at startup.** The idiom is the Market
  Journal's, unchanged: a `_loaded_once` flag and a `showEvent` override that calls the
  page's own reload. A `QTabWidget` child receives its `showEvent` only when its tab is
  selected, which is why `research_panel.py` needed no change at all - the tab widget
  already had the information, and the children were the ones not asking.
- **Deferring is not skipping, and the cost has to be visible where it lands.** Every
  deferred read still happens the first time the page is SHOWN. The re-measure shows that
  honestly: `research.construct` fell 3,025.1 -> 533.4 ms sync p95 and
  `research.tab.Day Trade Tracker` rose 2.3 -> 448.8 ms, because the Day-trade page's
  `reload_from_disk()` is still synchronous on the Qt thread. G7 moved WHEN it runs, not
  where; a packet that claimed the saving without naming where it went would be reporting
  a number rather than a measurement.
- **What a first-show gate must NOT swallow.** Market Prep's `QFileSystemWatcher` is still
  configured in the constructor and Price Alerts' status line is still rendered there,
  because neither reads a file and a page that only learned about a new scan once it had
  been opened would be a behaviour change. The Price Alert service's own monitoring timer
  is untouched: it is a separate question.
- **One reader per export.** The Setup Playbook's banner reads the short-horizon and
  recent-type exports through `setup_tracker_panel._load_csv_rows_cached`, the same reader
  the tracker page uses, so the two parse each file once per version instead of twice per
  render. `render_all_docs_html` and `render_best_now_html` take what they render as
  arguments; a renderer that opens a file is a render on the Qt thread waiting to happen.
- **A refresh that leaves the Qt thread has to say when it landed.** `refresh()` returns
  as soon as it has started a `ReadWorker`, so `refreshFinished` is the moment the rows
  are APPLIED. It carries no payload and the desk uses it for nothing; it exists because
  a test that cannot see the render can only assert on a call that no longer means
  anything.
- **The coalesced pass belongs INSIDE the worker, not in the Qt-thread slot.** A refresh
  asked for while one is in flight is taken by the worker in flight as one more pass,
  whatever the number of requests. Had the extra pass been started from the render slot,
  a request made a moment before the desk closed would leave a read starting after
  `shutdown()` had joined - the join would be joining the wrong thing. With the loop in
  the worker, joining the thread joins everything it was going to do.
- **A memo over the thing a table was BUILT from.** `_table_render_plan` names, per table,
  the export signature (and `min_closed` for the two tables the spinbox re-ranks, and a
  content digest for the human-focus table, which has no file). Two tables read the
  tier-performance export and both say so, so a rewrite of it re-fits both and nothing
  else. A cache that skips the parse but resets the model is not a cache the trader can
  feel.
- **A measuring tool reports its own overhead rather than assuming it away.**
  `_WorkerProbe` walks the candidate set once per op, re-walks it at most every
  `WORKER_REWALK_MS` (250 ms) while the page is busy, and re-walks it ONCE MORE before a
  settle is declared - the only moment a read that started after the last walk could be
  missed, which is exactly the Day-trade Tracker's second worker. `settle` returns
  `poll_cost_ms` as a fourth value and every op row carries it. On the after-run the worst
  poll cost was 91.5 ms, inside the 9.4 s `weekend.refresh_everything` settle: under 1 %,
  and now a number instead of a claim.
- **A test that calls a slot directly while the real worker is in flight is testing two
  writers.** `tests/test_g7_speed_pass.py`'s Market Journal test builds its capture in
  memory, so the panel's own `_CaptureWorker` lands an EMPTY payload for the same entry
  and clears the charts - green alone, red under load, purely on which arrived first. The
  test now lets the empty landing arrive first. In the real desk `_render_capture` is only
  ever called by that worker, once per selection, so this is a test artifact and not a
  behaviour to design around.

## The sentences CLAUDE.md moved here on 2026-09-07 (G7 docs pass)

`CLAUDE.md`'s size rule is that a rule there is one to three sentences naming its
seam, and that its story, numbers and quotes live here. It had drifted to 51.7 KB
against its ~45 KB limit, so the sixteen longest bullets were shortened and each
one's ORIGINAL text is reproduced below, verbatim and unedited. **Nothing was
deleted.** Where a bullet below and the current `CLAUDE.md` differ, the shortened
rule in `CLAUDE.md` is the binding one and this is the full account behind it; the
incident sections elsewhere in this file remain the deeper record.

### M4 - both AVWAP band families in one snapshot, long form

- **The daily snapshot carries BOTH AVWAP band families and they never share a column** (packet M4, 2026-09-05, BD-102). `avwape_*` is the champion (frozen, decision 0008); `avwap_variant_*` + `avwap_variant_formula_version` is the challenger, computed from the SAME bars and anchor index by `indicators.avwap_band_variants` and **independently of whether the champion produced bands** - a NULL band is "not measured", never a band on the centre line. `FEATURE_SET_VERSION` is `tier1_v2` and `tier1_v1` rows are never rewritten, so a session can hold one of each and **readers keep the newest `computed_at`**. `swing_house_variant_v1` is a `dataclasses.replace` twin of `swing_house_v1` differing ONLY in `band_family`; `build_outcomes` picks the band map from the RECIPE, and a variant recipe with no challenger bands walks `plain_no_target` rather than borrowing the champion's levels. Its `outcome_definition_id` (`band_variant_v1`) fences it out of every `house_default_v1` reader. `band-coverage --compare A B` pairs the two on the SAME occurrence ids with `swing_headline`'s ONE Wilson and counts an unpaired occurrence rather than dropping it. Shadow only; T4's criteria decide and nothing here promotes.

### N3 - the bounded narration view, long form

- **The `setup_research` narration is a BOUNDED view and its selection is a SIZE rule, never a ranking by result** (N3, 2026-09-05, BD-101). Every eligible cell died at 619 cells / 658,292 chars against a 78,119-char budget, and no 64k-context model reaches that, so "raise the budget" is not the fix. `_bounded_narration_view` encodes the fixed head first, then eligible policy cells by **`stats.n` descending, then `recipe_id`/`family`/`side`** until the next would cross the budget, then the after-like ELIGIBLE cells (P10 C3) by their top-level `n_episodes`. **No `mean_r`, `win_rate`, `profit_factor`, `expectancy` or any R statistic may enter that key** - gate #43 is a refusal. `narrated K of N` is stated in the view, the `.narration.json`, the pack markdown and the ledger reason; the refusal survives, narrowed to "the head plus the first cell does not fit", and names the head's size.

### P9 / N1 - the two like modes and the aware sidecar read, long form

- **A LIKE has two modes and only one names a setup** (P9). **Alt+L** is the QUICK like (`like_mode: "quick"`, no claim, no why, never prompts); **Alt+K** is the CLAIMED like (the claim is the whole gesture; the why is optional since T2). The quick-like BUTTONS prompt for an optional note; the key never does. `like_mode` is additive and its absence reads `claimed`. A like carries zero privileges, grades under `like_unclaimed` when quick, and contributes a LINK to the auto-tagger, never a tag. `sidecar_completion` finishes a capture sidecar into a NEW file (`m5_bars_completed_ref`); `m5_bars_ref` is never rewritten. **That read is AWARE (N1, 2026-09-05)**: a sidecar's naive `dt` is DESK-local wall time, so `pass_bars.desk_zone()` is ATTACHED to it (never an offset stripped), the close is 16:00 MARKET-local, and a new sidecar is written WITH its offset (schema stays 1); `lake_read_failed: <Exc>` is a READ fault and `research_store_unreachable` is `ResearchStore.open()` only - the naive bounds raised `ArrowInvalid` and read as "unreachable" every night from 2026-09-02 while the store answered 60 rows.

### M2 - a swept trade counts under the policy that measured it, long form

- A sweep-finalized trade counts under the policy that MEASURED it: `setup_scoreboard.exit_policy_r` keeps `eod_hold` / `stop_exit` / `last_measured` as separate columns, never blended; every eod-hold view reads `r_eod_hold`. **`unresolved` means UNMEASURED** (M2, 2026-09-05): a swept trade that measured its bars is written `swept_measured` and counts under the policy that measured it; every reader goes through `outcome_semantics.terminal_kind` (a status-less `final` with a numeric close is `measured_eod`, never `open`); the champion's eod-hold tier cells still take `eod_complete` only, BY DECISION (2026-09-06): a swept trade was measured under another policy and 99% of them sit in the freeze window, so they stay readable under `sweep_exit_policy_rows` and never enter the eod-hold cell - pinned by a golden characterization in `tests/test_setup_scoreboard.py`.

### The personal-evidence reader rules, long form

- **`preference_trade_outcomes` matches inside 10 SESSIONS** (`statement_window_end`; a calendar refusal falls back NARROWER) **and `trade_level_summary` sums P&L once per `trade_id`** over a file that stays one row per statement. **`journal_exposure` reads bias from the legs — a LONG option is never a bullish setup**, a `trade_legs` row is a FILL so `multi_leg` means more than one option CONTRACT, and a spread look-alike is `partial_of_spread_candidate` (no sibling seam exists). **`personal_evidence_summary` partitions by STATUS** — complete / partly_closed / open_exposure, whose net and winners are `None` — **with `uncertain` a CROSS-CUTTING label that pools no money**, counts both tag lanes over ONE denominator (closed or partly closed) and names no best setup below `MIN_REPORTABLE_N`; Weekend Prep lists the WHOLE provisional backlog and REFERS a missing `planned_risk` to the Trades tab, never computing one from an outcome.

- **5A - the verdict card reads NUMBERS, not strings** (WS-5A, 2026-09-12; WISHLIST item 5 block A, reproduced by Codex 2026-09-09). `weekend_verdict.best_cohort_line` composed its column name out of its horizon (`avg_r_h3`, `n_h3`) and the two panel readers that feed it published `horizon` / `n` / `avg_return`, where `avg_return` was already the table's formatted `+1.23%`. No row on either side of that seam has ever carried an `avg_r_*` key, so every cell was skipped and BOTH cohort lines printed "nothing with enough behind it yet" on a desk holding 115 graded veto rows and 129 graded like rows (30 and 33 of them at the three-session horizon) - and the first row that had matched would have printed a PERCENT return with an `R` after it. The contract is now typed at the reader: `_cohort_numeric_fields` gives `_read_veto_cohort` / `_read_like_cohort` a row carrying `horizon_sessions: int`, `n: int` and `avg_side_return_pct: float | None` (the CSV's fraction times 100; the unit is in the NAME because the bug was a unit bug), a blank stays `None` and never a substituted zero, and the `+1.90%` / `21` cells are built at the display edge by `_cohort_cell_text` inside `_fill_cohort_table`. `_cohort_view` compares the horizon as an int in one place, because `"3" == 3` is False and that is the class of mistake this repaired. In the card, `CARD_HORIZON` is the integer **3 sessions**, the pooled side (`ALL` is what the rollup writes; `BOTH` is carried too) is EXCLUDED from a ranking of reasons because it is both sides added together, and both lines rank by the HIGHEST side-adjusted return: "Likes that work: `<cohort> <side>` +x.xx% over 3 sessions (n=..)" and "Rejections worth another look: `<reason> <side>` +x.xx% side-adjusted (n=..)". The old veto line took `min()`, which named the rejection that was RIGHT - the one reading a trader never has to act on - so "Rejections that were right" is not printed at all: these are lines five and six of the eight the trader capped the card at. THREE absences, three sentences, because printing one for another is the class of false statement this repaired: under the floor is "nothing with enough behind it yet (best n was N against a floor of F)", graded only at other horizons is "nothing has matured to 3 sessions yet (K row(s) at other horizons)", and no rows at all is "no like|veto cohorts measured yet". The floor stays the card's own `MIN_COHORT_N` (5) rather than `evidence_stats.MIN_REPORTABLE_N` (30) by decision, because the card is a pointer at a table row and 30 would silence the like line entirely - 2 of the 21 three-session side rows clear 30, against 8 that clear 5; moving it is a trader decision about what the card may say, not a repair. Tests: `tests/test_ws_5a_weekend_verdict.py` (real live header, both sides plus the pooled one, a mature and a thin cell, an empty `avg_side_return`, and a characterization guard on the table cell text).

- **5B - the said-vs-did record is SYMMETRIC and the two families never pool** (WS-5B, 2026-09-13; WISHLIST item 5 block B). `collect_statements` asked `load_annotations` for `like_claim` and `pass` only, kept `verdict == "like"` from `pick_feedback` and had never opened the review-event store, so the report could say *"you liked it and did not take it"* and could never say *"you vetoed it and took it anyway"* - the half of the record that costs money. The reject half was missing by OMISSION, not by decision: nothing ever asked for it. Five reject channels now stand beside the four endorse ones and each keeps its own name, because no two verdicts are combined (P5): `annotation:veto` (statement_detail `<code> (v<vocab_version>)`, because cohort identity on write is that pair; an uncoded veto reads `uncoded` and borrows no version), `annotation:pass`, `pick_feedback:dislike`, `pick_feedback:not_today` (narrower than a dislike - one session thrown back, not the name) and `review_event:m5_click_away` (a click away from an M5 alert IS a pass, trader 2026-09-01; the row is `action: "skip"` with detail reason `clicked_away_from_m5_alert`, never renamed). **`unfavorite` is absent BY DECISION** and is in neither family. Three columns AT THE END of `COLUMNS`, schema `preference_trade_outcomes_v2`, the published nineteen byte-identical because `ai_summary.preference_to_trade_section` reads `match_basis`, `trade_id` and `session_date` out of this CSV by name: `like_mode` (read through `ui.annotations.store.like_mode_of`, so "absence means `claimed`" lives in ONE place - carried only by `annotation:like_claim`, because a ★ on a board and a swing favorite were neither the Alt+L key nor the claim dialog and naming either would describe a keypress that never happened), `verdict_family` (`endorse` / `reject`, from `REJECT_CHANNELS`; a row read back with the column blank is an endorsement, which every pre-5B channel was), and `match_state`. **`match_state` has four emitted values and a fifth reserved one**: `matched`, `window_open`, `no_match_after_window`, `journal_unavailable` - and `matching_unavailable`, which is in the vocabulary and which **no path emits today**, because nothing here can tell "the matcher could not run" apart from "the journal could not be read" and inventing a path to say so would be inventing evidence. The column AGREES with the AI section's derived buckets by construction: an empty `match_basis` is the `journal_unavailable` bucket there and here, and `window_open` is the same `statement_window_end(said) > reference` comparison, so the desk never holds two truths about one row. An unreadable journal no longer returns `skipped` with no file written - the trader still SAID it, so the statements are published with an empty `match_basis` and the slot reports `degraded` naming the journal. One thing said once is ONE statement: `_statement_identity` de-duplicates on the store's own event id where there is one (the annotation log heals torn tails rather than claiming atomicity) and on the statement itself where there is not. Money is still summed once per `trade_id`; `n_statements_by_family` carries BOTH keys always. `match_trade`'s side logic is UNTOUCHED (lead ruling): an opposite-side trade keeps matching at 0.35 with basis `symbol+window_opposite_side`, and a sideless coincidence stays `symbol+window_side_unknown` at 0.50 with `match_state=matched` - the ambiguity is labelled, never resolved. A dislike's free-text reason rides in `statement_detail` for a person to read and is never machine-coded into a cohort; it is not one of `ai_summary.PREFERENCE_EXAMPLE_COLUMNS`, so it reaches no model. Weekend Prep's Focus Review page gains a TENTH view, "Said no" (`preference_rejection_table` / `preference_rejection_note`), filled by the same single read pass, with `match_state` on screen because "no trade yet" and "no trade, window closed" are a pending row and a broken promise. Tests: `tests/test_ws_5b_preference_symmetric.py`.

### ST4 / ST7 - the named selection policy, long form

- **Which observation of a thesis gets graded is a NAMED policy** (ST4/ST7): family rows carry `selection_policy`; `first_actionable_v2` (`master_avwap_lib/selection_policy.py` - earliest row per attempt, declared re-entry rule, declared `full_band2` representative, pending stays pending) is the default since 2026-09-06 (decision 0019), `closed_first_v1` remains by name, and the persisted write logs `policies: selection=.. execution=.. levels=..`. A scenario's exit date is its last `events` entry, read only when CLOSED; a COMPACTED record is `undatable_exit`, never silently pending. **A compact projection's `_scoring_outcome_summary` IS the record**: a default read takes it unconditionally and an ABSENT `representative_status` grades off `closed_setups` - recomputing dropped every setup and zeroed both deltas.

### ST6 - the one Working-lately snapshot, long form

- **ONE Working-lately snapshot, four surfaces, and nothing called proven** (ST6): `working_lately.build_snapshot` is PURE and `snapshot_id` is a sha1 over the sorted cells + the declared policy lines + `as_of` alone, so a timer tick cannot move it; `ui/services/working_lately_service.py` owns the worker build, `snapshot_latest.json` and the deduplicated `leader_change_events.jsonl` (cause REFUSAL-FIRST: `lost_coverage`, `corrected_data`, `window_rollover`, `new_outcomes`); the strip, the tracker banner (its own read labelled `panel read`), Weekend Prep and the AWAY Recap render THAT payload. Dependence is answered by REFUSING - a cell over `CONCENTRATION_LIMIT` (0.5) of one name or session cannot lead, `pool_cells` RAISES across kind/side/outcome kind, `LEADER_PERSISTENCE_SNAPSHOTS` (2, declared 2026-09-06 before any forward look) holds a NEW leader for two distinct-`as_of` snapshots, and every surface prints `observational leader among K cells`.

### Headline statistics - the ST1/ST2 clauses, long form

- **Win rate leads every trader-facing SWING surface** (first, with `n` and a Wilson lower bound from `swing_headline`, sorting by the bound, mean R beside it); **MFE-after-a-held-level leads every DAY-TRADE surface**. ONE WILSON: `swing_headline`'s z (1.96); `expected_r.py`'s 1.28 is a parameter inside a fenced file. **Counts are integers at each table's own grain** (ST2): the recent, Setup Types and short-horizon exports carry `n_wins`/`n_losses`/`n_flats`/`n_unmeasured`/`n_pending`, the weighted rate keeps its own name (`win_rate_closed_basis`), a count is never rebuilt from a rate, and a row without them says `counts not exported yet`. **ONE leader through `working_lately.select_leader`** (declared `LEADER_MARGIN_LB` 0.05 and `LEADER_FRESHNESS_SESSIONS` 2, four states, the floor judged before the clock, a study never leads, a `discovery_leader` is never called one); a page computes its verdicts ONCE (`panel_verdicts`) and every renderer shows all four states - a hardcoded fallback sentence is a state the machine does not know. **The tier outcomes `win` is a FAVORABLE-DIRECTION flag at a scan-row offset, not a stop-rule win** (ST1): `outcome_kind` says so, `Headline.outcome_kind` makes those surfaces say "favorable", `swing_evidence.read_eligible_rows` is the ONE reader, and the exact-session v2 rows (`session_horizon_outcomes.py`) are a separate shadow file with no production caller.

### Q1 - the day-trade headline and measured held, long form

- **The day-trade headline is `held_run_score`**: P(held in the first 30 min) × trimmed-mean MFE_R of the held ones; ONE formula reaches every surface (the tracker joins `dimension_summaries`; the M5 row reads `alert_cell` + `alert_suffix`); segments are spelled the AGGREGATOR's way; `UNDERIVED_DIMENSIONS` separates "the log cannot answer" from "reachable, not derived". **Held is MEASURED held** (Q1): `measured_held` / `measured_broken` / `pending` / `unmeasured`, `hold_rate` = held / MEASURED, the unmeasured counted and shown, never assumed; the D1 dimension is the ALIGNED same-session setup from `master_avwap_tracker_scoring_snapshot.json` (index expires on the day roll), missing snapshot = UNKNOWN. My Decisions rows use `ALL_DIRECTIONS`, a pooled cell accumulated from the episodes, never an average of two cells. The window is `evidence_stats.lately_window`, gaps reported.

### T1 - the four capture verbs, long form

- **A VETO retires the chart, a CLAIMED like ADVANCES it, a QUICK like and a NOTE move nothing** (trader, 2026-09-04). A rail veto uses its own verb (`vetoRetireRequested` → `_retire_after_veto`) and writes ONE row; the "✕ Not today" button writes an uncoded row and opens the note box. A quick like is `likeRecorded` → `_after_like`; a claimed like is `likeAdvanceRequested` → `_advance_after_like` → `_advance_review_queue`; an advance parks nothing and drops nothing. Both write `like_advance` through `_record_like_advance` because `review_learning.TAKE_ACTIONS` keys on it. "Veto D1 — but M5 today" writes a veto row and emits a REQUEST; the panel places (first), then retires (second) through the box-free verb.

### The four auto-tagging lanes, long form

- Auto-tagging has four lanes that never compete and are ordered by LANE, never confidence: `trader_capture` (what the trader already SAID inside the trade's own window, outranking every other source, a rejection prefixed `vetoed:` / `passed:`), then `trader_note` (WS-10E's Market Journal lane), then `journal_analytics.AutoTagger`'s scanner-file lane (which setup, from the scanner's own files), then `journal_trade_shape` (facts from the trade's own timestamps and legs). No tag is ever derived from the outcome; unmeasurable emits NO tag; `context_row_id` is a pointer, and plan.md P5.3/P5.4 own the canonical opportunity id. `preference_trade_outcomes` shows its match confidence on every row or says "no match".

**WS-10E - the note lane** (WISHLIST 10E, built 2026-09-13). The Market Journal has held
the trader's own written setup claims since R10.H and no lane read one: `grep
market_journal scripts/journal_analytics.py` returned nothing. A sentence typed about a
name while the trade was open reached the Journal's Tags column by no path at all.

- **The window is the trade's OWN, with one trading session of margin before the open.**
  `AutoTagger.note_window_for` walks `market_calendar.previous_session`; the write time
  compared against it is the entry's `created_at`, because the subject session and the exact
  write time answer different questions. A
  **date-only broker fill has no intraday window** (`journal_trade_shape.is_date_only`) and
  the verdict is `unmeasured`, never a tag.
- **The lane matches a CLAIM, never a mood.** The vocabulary is `setup_docs.SETUP_DOCS` -
  the encyclopedia the desk already keeps - compiled into whole-token phrase patterns from
  each family's key and its label. A single-token phrase is DROPPED, so `general` can never
  tag a trade because the trader wrote "general weakness". The candidate carries
  `match_basis = note:<entry_id>` and the `span` quoted verbatim out of the entry.
- **Side comes from the note's OWN words.** A note whose words state the opposite side
  never matches (matching on the ticker alone is exactly what the rule forbids); a
  side-silent note matches either, because silence is not a contradiction.
- **Confidence 0.88 / 0.84** - below the capture lane's 0.90/0.95 and above the scanner
  lane's observed ceiling (P6a's histogram: tracker + same day + same side is 0.72) - so
  `journal_bulk_tag` picks it up under the same 0.70 threshold with no change to its pick
  rule. `JournalStore.apply_provisional_tags`' refusal to overwrite a confirmed tag is
  untouched and pinned by a test.
- **THREE verdicts and no fourth**, stored on the trade as `note_lane_json` by every
  `refresh_auto_tags` and printed by `journal_analytics.format_note_lane_line`: `note lane:
  <setup> from note <id> "<span>"`, `note lane: no explicit claim in N candidate note(s)`,
  `note lane: unmeasured (date-only fill)`. Stored rather than re-derived because the
  Journal's Trades detail must not open the Market Journal ledger on the Qt thread.
- **Tag Week's `From` column is COMPARED, never flagged.** `note_lane_tag` against the
  row's own `setup_tags`, so the mark stops claiming a provenance the tag no longer has the
  moment the trader edits it, and a confirmed row is never marked.
- The advisory package gains `trader_notes` (id, market-local write time, text, side words)
  and `deterministic_note_lane`, so the model cites `note:<id>` in AI1's `sources` and is
  correcting an answer rather than inventing one. Nothing in that path writes a tag.
- Tests: `tests/test_ws_10e_note_tags.py` (13), including a grep-guard over every callable
  named `*note*` in `journal_analytics` proving the lane's inputs reach no outcome field.

### A broker file is authoritative for money, long form

- **A broker file is authoritative for money and blind to time.** `journal_statement_import` (Questrade `.xlsx`, no `openpyxl`) and `journal_ib_transactions` (IBKR sectioned csv, USD price / CAD money, masked accounts unmasked only when exactly one fits) write executions at MIDNIGHT market-local; `journal_trade_shape.is_date_only` refuses to name a session for them. Side and options come from the DESCRIPTION; identity is `fill_signature` plus an ordinal, never positional; a statement never writes into a (broker, account, day) a richer source already covers. Commission carries a SIGN and the importer owns it — nothing downstream may `abs()` it.

#### TJ-9Q - a Questrade fill says what it is, and a sold put is a sale (2026-09-19/20)

Branch `claude/tj9q-questrade-instrument` (tip `76f3cf2a`), **merged into
`lead/p033-integration2` as `86c64f96`** - not on `main`, and the stored rows have NOT
moved: `--apply` on the live journal is the trader's own act and is owed.

The trader sells puts; they do not buy premium (one recorded exception, a QQQ put bought
and sold on 2026-06-26). Until 2026-09-19 the journal said the opposite. Measured that day
and re-measured on 09-20, both on byte-exact COPIES of `trade_journal.sqlite3`:

| measurement | value |
| --- | --- |
| Questrade executions / `security_type = UNKNOWN` | 226 / 226 |
| payload keys per execution | exactly 20 - no `securityType`, no `symbolType` |
| `net_amount` / `gross_amount` on Questrade rows | NULL on all 226 |
| stored `multiplier` column | 1.0 on all 226, options included |
| stored sides | SELL 101, BUY 92, COV 25, STO 4, BTC 4 |
| open `UNKNOWN` Questrade positions | 29 (24 OPEN + 5 CLOSED_PARTIAL) |
| the four option positions | 10 fills, 4 symbols |

Two causes, both vocabulary. The executions endpoint states no type, so
`normalize_security_type(None)` answered `UNKNOWN` and `journal_identity` then priced a
contract at one. `normalize_side` knew `BTO` and `STC` but not `STO`, `BTC` or `COV`, so
those three were stored verbatim and `_signed_quantity` - which negates only its own SELL
set - read every one of them as a BUY: `STO` opened the position LONG, `BTC` added to it
instead of closing it, and `COV` came out right by accident. **Three sold-put positions**
(four STO fills - AAOI has two) read `direction = LONG` and sit OPEN with
`quantity_closed` 0, and the ONE bought put (QQQ, BTO/STC) is correctly LONG and CLOSED
but 100x too small. TJ-9's item 7 was refuted as an additive change on exactly this: a
forward-only classifier would split every open position from its closing fill, because
`group_key` and `trade_id` key on the type. `execution_uid` never moves, and
`fill_signature` is the CSV path's and is untouched.

**The answer is in the broker's own fields, twice over.** The symbol is Questrade's own
option spelling (`AAOI18Jun26P120.00`) and the side word is one of four.
`classify_questrade_security_type` requires BOTH to agree, which makes them a check on
each other rather than a guess; a payload that does state a type (`get_positions` does) is
believed over both; a disagreement, an unreadable symbol or an absent side stays
`UNKNOWN`, because an UNKNOWN position is visible and fixable and a wrong one is silent.
Nothing is inferred from a ticker's LENGTH. The stored SYMBOL may not move either:
`canonical_option_symbol` rewrites a symbol into OCC form as soon as the type is `OPT`,
and the root, expiry, strike and right are not in this payload - so the symbol is still
spelled from the type the payload STATED, and classification is additive.

**One journal, one convention.** With 29 positions open under the old spelling, a new fill
classified `OPT` could not close a position grouped `UNKNOWN` - it would open a second
one. So the classifier and the side map are pure and always correct, and the SEAMS consult
`QUESTRADE_INSTRUMENT_FROM_SYMBOL` (ships `False`; the effective value is read at CALL
time from `local_settings["questrade_instrument_from_symbol"]`, absent = False): the
importer's `_append_normalized` and `journal_statement_import._execution_from_row`, never
the classifier. With it OFF both paths are BYTE-IDENTICAL to before - raw side word
stored, `UNKNOWN`, multiplier 1.0, `STO`/`BTC`/`Cov` CSV rows still dropped - and the
reviewer proved that base-versus-tip through the real seams. The old map survives by name
as `normalize_side_pre_tj9q` for exactly those gated seams.

**What the correction is worth**, measured on a copy of the live journal:

| position | before | after |
| --- | --- | --- |
| AAOI 18Jun26 P120 | LONG OPEN, qty 4/0, -3.965665 | SHORT CLOSED, 2/2, **+241.034335** |
| BE 2Jul26 P260 | LONG OPEN, 2/0, -1.987107 | SHORT CLOSED, 1/1, **-666.987107** |
| QBTS 26Jun26 P22.50 | LONG OPEN, 4/0, -3.961772 | SHORT CLOSED, 2/2, **-57.961772** |
| QQQ 26Jun26 P700 | LONG CLOSED, -2.782369 | LONG CLOSED, **-81.982369** |

Every one equals the broker's own `totalCost` arithmetic to the cent - and `totalCost`
already carries the multiplier, which is what makes it an independent check rather than a
restatement of ours. Total recomputed P&L across all trades 4,737.45 -> 4,184.25: the
-553.20 is exactly the four option corrections, and no equity trade moves.

**The stored rows move once, through a tool that refuses.**
`scripts/journal_reclassify.py` (`--db`, `--dry-run`, `--apply`, `--i-am-the-trader`,
`--verbose`) is a dry run BY DEFAULT and the dry run reads COPIES - it does not open the
database it is reporting on, so it cannot change a byte of it. `--apply` takes a
byte-exact timestamped backup FIRST, moves type + side + multiplier together through the
ONE additive `JournalStore.reclassify_executions` (Questrade rows only - it refuses any
other broker's uid, and the SQL carries `AND broker = 'QUESTRADE'`), rebuilds, and
verifies: the same executions, every broker-stated amount unchanged, and no newly stranded
annotation or machine row. A failure RESTORES the backup byte-exact and exits 1. The three
fields travel together because they are read together - `group_key` reads the type,
`_signed_quantity` the side, and `journal_file_authority._multiplier_for` the stored
multiplier COLUMN before it will look at the type.

**A mass re-key must carry every trade-keyed table, not only the trader's.** Each
`trade_annotations` row is re-keyed by largest execution overlap and a TIE is REFUSED,
never guessed: a contract sold, bought back, sold again and bought back again is one open
position today and two closed round trips afterwards, each holding half the executions -
that position is left entirely in the old convention and NAMED. The machine's rows follow
the same rule, each table by its own nature: `ai_trade_enrichment` (nothing regenerates
it) refuses its POSITION when it cannot be placed; `note_lane_verdicts` (derived, and
`refresh_auto_tags` already deletes dead ones) is DROPPED, counted and named;
`opportunity_events` is NEVER rewritten - it is carried by a `trade_aliases` row, and
unresolvable references are counted and named.

**The switch is written LAST, and never onto a half-moved journal** (the round-1 blocker:
it flipped ON after a refusal, and a new fill then split that contract into two
positions). `--apply` writes `local_settings["questrade_instrument_from_symbol"]` only
when NOTHING was refused AND no open `UNKNOWN` Questrade position is left; otherwise
everything that moved stays moved and correct, the blockers are named, and the run exits
**4**. Exit codes: 0 done · 1 verify failed and restored · 2 would not start · 3 busy (the
desk slot or the `ai_jobs_runner` lock - the nightly journal import runs inside that
runner) · 4 rows moved, switch stayed off, re-runnable. `--apply` refuses a `--db` under
`C:\TradingBotData` or the DAS without `--i-am-the-trader`. If the power goes out
mid-apply, running `--apply` again repairs a reclassified-but-not-rebuilt journal.

**Measured by the reviewer on a copy of the live journal, 2026-09-20:** dry run 0.2 s and
zero bytes changed; `--apply` 0.3-0.4 s; 226 fills moved, 0 refused; `security_type` moved
on 226 (OPT 10 / STK 216), `side` on 33 (STO->SELL 4, BTC->BUY 4, COV->BUY 25),
`multiplier` on 10; ZERO non-Questrade rows touched; all 616 `execution_uid`s and every
`net_amount` / `gross_amount` / `commission` / `fees` / `quantity` / `price` byte-identical;
exactly FOUR positions move; 185 annotations in and 185 out on live trades, content
identical, `label_provenance` still present and EMPTY on all 185 (a maintenance pass may
not invent a date for a call the trader made); `ai_trade_enrichment` 24 of 24 on live
trades; `note_lane_verdicts` 94 carried and 4 dropped (CRDO, FBIN, IOVA, SMH - one
execution shared by two rebuilt trades; machine `no_claim` verdicts, nothing the trader
wrote); every existing `opportunity_events` row byte-identical, 94 aliases, 4 left and
named; the tax report identical for 2026 (76 positions, CAD 5,426.33) and 2025 (24, CAD
1,308.97), because it sums `net_amount`, which is NULL on every Questrade row and is never
written here - what changes is an exclusion REASON, "still open" becoming "a fill carries
no broker-stated amount", a better sentence about the same zero dollars. A second
`--apply` is a no-op. Page one of the report is ~40 lines, decision first (the builder cut
it from 609).

**The ONE ungated change, lead-approved:** `journal_file_authority._BUY_SIDES` gains `COV`
(the set held `COVER`), so the trader's 25 covers stop counting as cash coming IN. It
moves 14 (account, day) pairs, 72,596.51 in total absolute = twice the covers' 36,298.26
(SMPL 2026-09-18 alone +366.98 -> -366.98), and it makes the file AGREE with the sync on
those days. It has no automatic caller - only a trader-initiated statement import or
"Check a statement..." - so it lands when this build reaches the desk, not at `--apply`,
and the dry run says so rather than leaving it out of its before/after. Also ungated and
correct: a hand-typed `STO` in the manual-fill dialog now stores SELL.

**The trader's own steps, in plain words.** By DAY with the market closed, never between
22:00 and 06:00 Pacific (the overnight AI jobs run then, and the nightly journal import
runs inside them). 1. Close the Trading Desk. 2. Look first, changing nothing:
`.venv\Scripts\python.exe scripts\journal_reclassify.py --db "C:\TradingBotData\data\runtime\trade_journal.sqlite3"`.
3. Read page one: AAOI goes from `LONG OPEN -3.97` to `SHORT CLOSED 241.03`, BE to
-666.99, QBTS to -57.96, QQQ to -81.98; `refused: 0`; and under **THE SWITCH**,
`ON.  --apply would turn it on.` If `refused:` is not 0 or the switch says OFF, it is
refusing to guess about something the trader wrote - send that page back before going on.
4. Run it for real with `--apply --i-am-the-trader` on the same command. 5. Read the SAME
two lines on THAT output: `refused: 0`, and the switch line must read **ON. New Questrade
fills are stored the same way from now on.** 6. Keep the printed backup for a week.
7. Start the desk and check the three puts read SHORT and CLOSED. To undo: copy the backup
file back over `trade_journal.sqlite3`.

### M1 - the challenger measured through the catch-up path, long form

- **The AVWAP band challenger is measured through the CATCH-UP path too** (M1, 2026-09-05): `build_anchor_band_variant_meta` lives in `legacy.py` and serves BOTH the live scan (`runner.py` re-exports it) and the tracker catch-up (`_evaluate_priority_snapshot_for_date`), which never set the block, so the shadow measured nothing from 08-26 to 09-05. Never add a third builder; a record is rebuilt on every persisted tracker write, so no migration exists. The Band variant tab prints `Measured N of M setups (K unmeasured: <top reason>).` from the export's own counts (`top_unmeasured_reason`), never by reading the 1.1 GB tracker; still shadow only, T4's 20-session accrual starts at the first measured row.

### M5 - the control, study and exit-framework populations, long form

- **The control, study and experimental-exit populations are SURFACED, LABELLED, and never mixed with picks** (M5, 2026-09-05): three Setup Tracker tabs - Controls (`N graded episodes from the M setups`, never one number under the other noun), Studies, Exit frameworks (`comparison_apr2026` beside `baseline`, `n_filtered_by_experiment` reconciling the two n's) - read three CSVs written in the tracker's own guarded save pass. Win rate leads with `n` and the ONE Wilson bound, the sort is the bound, each tab carries a population sentence and `experimental` is a COLUMN. Shadow only; the champion aggregates are pinned byte-identical.

**EF1 - the same comparison, split by setup family** (trader, 2026-09-08; built in the
2026-09-12 WISHLIST sweep). The trader asked whether taking profit at the 3rd band beats
the 2nd for the 1st-dev breakout study, and the M5 export could not answer it: it pools
every scan row by `(framework_family, exit_template_id, side, priority_bucket)` and never
by SETUP, so "full at band 3 loses" (LONG favorite: 47% win, -0.15 R, n_closed 8,018) is a
whole-population answer. A 1st-dev breakout starts one band from its target; an AVWAPE
bounce starts two.

- **The grouping key is a PARAMETER of the one builder, never a second builder.**
  `legacy.build_exit_framework_stats_rows(setups, by_family=False)`; with `by_family=True`
  the key gains `setup_family` and `population` in front and the rows carry those two
  columns first (`EXIT_FRAMEWORK_BY_FAMILY_STATS_COLUMNS`, derived from the pooled tuple so
  the two files can never disagree on a column, and pinned that way by a test). One builder
  is what makes the two files agree on a rate; one scenario walker
  (`_flatten_tracker_scenarios`) is what keeps the band-variant fence - a second walk of
  `setup["scenarios"]` here would be the eighth unfenced reader
  `test_band_variant_fence_guard.py` exists to prevent.
- **A SECOND file, never a finer grain inside the shipped one.**
  `master_avwap_exit_framework_by_family.csv` beside `master_avwap_exit_framework_stats.csv`,
  written in the same guarded save pass under its OWN `try`, after the pooled one. The
  pooled file stays byte-identical (golden) and a raising by-family export costs neither
  the tracker save nor the pooled file. The pooled file's 24 live rows sit under the
  table's 300-row cap; splitting it by family would blow the cap and change the grain of a
  shipped table.
- **`population` is the RECORD's flag, joined on `setup_id`, never the family name.**
  `champion` / `study` / `control` from `is_study` / `is_control`, so a champion family
  called `study_1stdev_breakout_probe` is still `champion` and a study family with no such
  word is still `study`; population is part of the key, so a study and a champion sharing a
  family name keep separate rows. The by-family export reads all three namespaces (the
  study and control records carry the same exit scenarios, built by the same
  `build_tracker_setup_record`); the pooled export still reads `setups` alone. So the
  reconciliation - family sums of `n`, `n_closed`, `wins`, `losses`,
  `n_expired_unmeasured`, `n_filtered_by_experiment` equal to the pooled row - holds over
  the CHAMPION rows. A setup with no `setup_family` is `unlabelled`, counted, never dropped:
  a dropped row would make the pooled row bigger than the sum of its families with nothing
  on the page saying so.
- **The tab gains ONE control and nothing else.** `exit_framework_family_combo` above the
  population sentence, first entry `All setups (pooled)` rendering today's table unchanged
  (golden order), then every family in the by-family export sorted by NAME - never by a
  result, which would make the choice for the reader and move under them between scans. A
  family view FILTERS BEFORE the 300-row cap (sixty families of six rows is 360; a view
  capped before it filtered would show a late family nothing) and re-ranks nothing:
  `_rank_exit_frameworks` has already ordered both files by the same Wilson lower bound.
  The family sentence prints the LARGEST `n_closed` among the family's rows, never the sum -
  the four templates are simulated on the SAME setups, so 30+33+31+32 is a claim about 126
  setups that do not exist - and says `BELOW FLOOR` when every row is under
  `evidence_stats.MIN_REPORTABLE_N`, with the rows still SHOWN. Hiding a study's rows is
  how a study never gets looked at, and a study is what the split was built to read.
- **The table keeps its OWN render memo** (`_exit_framework_rendered_from`) rather than an
  entry in `_rendered_from`: that dict is replaced wholesale at the end of every render
  pass, so a key written into it there would be dropped and the table would re-fit on every
  refresh - the defect G7.2 measured and fixed. A picker click re-renders one table from
  rows already in memory and reads no file.
- **Shadow only.** No detector, score, alert, template, stop rule or default exit changes;
  nothing here promotes a template - T4's criteria decide. File-scoped ask-first: the
  trader's EF1 prompt is the yes for the exit-framework export seam only.
- **Tests:** `tests/test_ws_ef1_exit_by_family.py` (19; 18 failed with the fix reverted,
  proved 2026-09-12).


### The M5 Strength Board's auto-adoption, long form

- **M5 Strength Board:** batched yfinance over `universe_all.txt` PLUS the four trader watchlists, zero IB traffic; relative volume is SESSION-RELATIVE and is not one of the seven fenced formula functions (byte-identical to the R8 baseline); D1 SMA floors read `2y` with today's forming bar dropped. Its parity rows auto-join M5 Focus (`_auto_adopt_strength_board`: DESK only, empty `failed_floors` only, the ONE adoption gate re-run per row, skipping `_ignored_symbols` and `FocusPickStore.declined_today`, one `add_many` per side plus `mark_auto_adopted`, never `FocusService.add`, never removing). Every Focus add is injected into `longs.txt` / `shorts.txt` by `FocusPickStore._inject_into_shared` and a removal un-injects it.
- **WS-10B (2026-09-12) — the trace, and the `Scan` column.** WISHLIST 10B asked first for a *verification*: follow a board row to the universe BounceBot actually scans. Traced and PINNED (`tests/test_ws_10b_board_to_scan.py`), the DESK link was intact end to end — board → empty `failed_floors` → the ONE adoption gate → `store.add_many` → the `focus_auto_picks.json` marker → `_inject_into_shared` (appends only when the name is absent, so a 15-minute refresh appends nothing and a trader-typed line is never touched) → `longs.txt` / `shorts.txt` → `BounceBot.get_scan_symbol_set`, which re-reads both files every cycle, so an adopted name is scanned on the NEXT cycle without a restart. **What was broken was AWAY**: the auto-mode matrix says AWAY stages and never adopts, and the board did neither — it returned at the mode check and threw the discovery away. AWAY now STAGES the eligible rows through the existing owner of the queue (`autopilot_core.stage_auto_populate_candidates`: one lock, one file, the per-side cap, a name already on a watchlist or already decided today skipped), so `_poll_auto_pick_queue`'s drain — which re-measures every queued pick on the flip back to DESK before adopting — stays the only door an unattended pick comes through. Nothing new polls; this rides `boardChanged` as the adoption already did. EVENING and OFF still do nothing. `gate_bar_end` is left EMPTY on a board-staged pick on purpose: an empty measured-bar stamp refuses at adoption (`pending_pick_gate_ok`), so the flip's re-verification is what admits it.
- **Every row now says why** (item 2). `_auto_adopt_strength_board` writes an `adoption` verdict per row at the moment it decides, from the numbers it decided on — `adopted` / `already_in_focus` / `staged (AWAY)` / `not today` / `declined today` / `mode EVENING` / `mode OFF` / `not adopted: floor <what it missed>` / `not adopted: <the adoption gate's reason, verbatim>` (an UNKNOWN bar therefore reads `not adopted: cannot verify session VWAP`) — and pushes it to the board through `StrengthBoardPanel.set_adoption`, keyed per SIDE because one symbol can sit on both tables with two different answers. It renders as the LAST column, **`Scan`**: text only, no colour vote, and **not sortable** (`SCAN_COLUMN`) — every other column is a measurement and ranking by one is the board's job, but a scan list re-ordered by how the machine answered is not the trader's ranking. The view computes NONE of it: a second opinion rendered beside the first is the disagreement the one gate exists to prevent. A row with no verdict is BLANK, never `not adopted` — the auto-join returns early when it cannot reach the Focus store, and a blank cell says "nothing decided this refresh" where a refusal would name something that never happened. One INFO line per refresh: `Strength board: N rows, A adopted, S staged, R not adopted (reasons: ... x<n>; ...)`, where R counts every row that ended neither adopted nor staged, so the four numbers add up.
- **Scanner inclusion and Focus adoption stay DISTINCT contracts.** Nothing here scans a row the gate refused. If the trader ever wants every board row in the scan set regardless of adoption, that is a separate selection and a separate decision — it was explicitly NOT built.

### Every ticker click lands on the centre chart, long form

- **Every ticker click on the Trading Desk charts into the centre Visual Alert Review pane** through `chart_symbol`, never `_enqueue_review_alert` (panels carry a `set_chart_sink` that `set_mode` points at `chart_symbol` in workspace mode). **A board chart holds NO place in the waiting list and is never re-queued or skip-counted** (`_is_manual_chart_look` on `MANUAL_CHART_TAG`); looking at a WAITING name takes it out for good. `show_board_symbol` is the popup door for a board on ANOTHER page. The RS/RW board and the M5 Strength Board were sections under the Desk's Strength window until 2026-09-07; since then they are blocks on its one flat page (next entry). One `StrengthBoardService` owned by `MainWindow`.

### The Strength window is one flat page, long form (trader, 2026-09-07)

- **What the trader said:** *"For the main trading desk, the strength tab is unusable there's like 2 tabs and they get no space each. Create a solution that removes the tabs and just collates all the data to be more easily readable."*
- **What was there.** The right-hand 40% of the Alert Center's lower row was a COLUMN: `FocusStrengthBoard` (stretch 1) over two `CollapsibleSection`s - "RS/RW Board" (the entry-assist board over the RRS snapshot, starting OPEN since V1 / decision 0016 answer 7, stretch 3 when open) and "M5 Strength Board (TC2000)" (starting closed since 2026-08-31, stretch 2 when open) - each section's body its own scroll area. That column is the lower third of one desk column, so with both sections open three documents shared a few hundred pixels of height: each got a window a fraction of its own height with its own scrollbar, and the closed one hid a whole read behind a header row. The trader's "2 tabs" are those two section headers.
- **What it is now** (`ui/widgets/strength_page.py`, `StrengthPage`): ONE `QScrollArea` with ONE scrollbar, the four reads one under another in the order the sections had - Focus strength, the auto RS/RW entry board, the RRS snapshot, then the M5 Strength Board under a `SectionTitle` heading the page owns (the panel never had one; the section named it). Every `QTextBrowser` is sized to its DOCUMENT by `fit_height_to_document` (a `_DocumentFit` helper parented to the browser: vertical bar off, `Fixed` vertical policy, re-fitted on `textChanged` and on the layout's `documentSizeChanged`, so a width change reflows and re-fits with no resize handling of its own), and every `_SideTable` to its ROWS by `set_fit_rows` (header + rows + frame, capped at `FIT_ROWS_CAP` = 30 because parity OFF is the top quarter of ~1,100 names per side; past the cap the table keeps every row and scrolls them itself - the cap bounds the height, never the rows). The RRS snapshot's three scopes are STACKED (`set_stacked_scopes`, `_board_html(..., stacked=True)`): three four-column tables abreast in ~490 px read at ~50 px a cell.
- **Measured while building:** a `QTextEdit`'s document has page width 0 until the widget's first relayout (show or resize), and `QTextDocument.size()` reads `(0, 0)` until then - so the first fit lays out at the viewport's width itself, and the first real relayout replaces that width through `documentSizeChanged`.
- **What it costs the charts: nothing.** The boards' minimum widths stop at the page's scroll area as they stopped at the two section scroll areas; the page's floor is the column's old 170 px, so 170 + the tab stack's 170 stays inside the alert column's 360 px budget (`test_the_page_keeps_the_alert_column_floor`). The page's vertical scrollbar is ALWAYS ON: an as-needed bar that appears takes ~16 px of document width, reflows every board, changes their heights and can make itself unnecessary again - a resize loop on the Qt thread. A bar that is always there costs 16 px once.
- **What did not change:** every widget, signal and owner. `MainWindow` owns the one `StrengthBoardService`; `attach_strength_board` hands the panel to `StrengthPage.attach_strength_board`; every ticker click still charts into the centre pane; the adoption gate, the parity toggle, the sort, the Copy RS/RW buttons and the two review buttons are untouched. `CollapsibleSection` has no caller left and stays as a widget. Pinned in `tests/test_qt_strength_board_in_the_desk.py` section 5 (seven tests, RED first).
- **File-scoped ask-first:** `alert_center_panel.py` houses alert code; the trader's message IS the instruction for this window, and the edit is hosting only - no alert, tier, fold, queue or evidence behaviour touched.

## SN - the scanner's hold on the interpreter (2026-09-08 through 2026-09-15; SN1-SN6 built)

- **What the trader said:** the desk was *"really quite laggy"* at the close; then, on 9% of the week's usage, *"Anything we can get done from wishlist.md that's quick and cheap?"* and *"Go"* for SN5 and SN6.
- **What was measured (2026-09-08, `thread_cpu.jsonl` / `ui_stalls.jsonl`):** `Thread-4 (run_strategy)` at 0.62 of a core on average in hour 13, 71-88% per minute at the close, climbing from 0.35 at 06:00; the GUI thread at 0.10-0.15; 13,031 GUI stalls over 50 ms (four hours hit the 2,000-record cap, so undercounted), 245-365 blocked seconds per hour, one 30.6 s stall at 13:01:53. Cycle 24's preamble was 658 s: the fast lane 333 s over 258-259 names (107 auto-adopted + 64 + 129 trader files, alphabetical), then four RRS passes 275 s. A cycle is ~25 min and `wait_for_candle_close` returns at once, so the loop never rests.
- **Why a thread cannot be made polite:** the scanner shared the interpreter lock with the GUI; a CPU-bound thread released it only at the interpreter's switch interval, and no priority trick freed it (the F1 lesson). SN1 now owns `run_bot_with_gui` in one Windows-spawned below-normal child (`ui/services/bounce_process.py`). Commands cross one pipe; every GUI callback crosses one bounded queue in original order. `BounceService` is the only owner, restarts a dead child on the health tick, and retires or terminates it on shutdown. The proxy exposes the existing chart/regime/entry-assist reads; chart ticks request one symbol rather than copying the full cache, while the warehouse retains its once-per-minute snapshot. The detector, bars, timing, alert filters and evidence writers are unchanged.
- **SN5 - the breath.** `BounceBot._breathe` waits `SYMBOL_BREATH_SECONDS` (0.02) on `_stop_event` after each symbol's compute, in the fast lane and in both main-sweep loops. On the stop event and never `time.sleep`, so a set event returns immediately and shutdown is not one symbol slower. Cost: ~586 symbol scans a cycle x 20 ms = ~12 s on a 25-minute cycle. Pacing only: the loop, the set, the bars and every output are unchanged; `ScanCycleClock` still never sleeps.
- **SN6 - the trader first.** `BounceBot._fast_lane_order` returns the trader's own Focus names (no marker in `focus_auto_picks.json`) alphabetically, then the auto-adopted ones alphabetically; the sweep follows as before. The engine reads the markers through `focus_picks.load_auto_pick_symbols()`, which shares `_read_todays_auto_pick_markers` with the store, so both apply the same per-entry `session_date` rule (R2.1). A failed or missing read is an EMPTY set: every name then scans as the trader's, which promotes and never demotes or drops a name - absence of a marker means the trader owns it (R2). The fast-lane log line prints both counts.
- **What did not change:** `request_and_detect_bounce`, the RRS passes, `_rebuild_feed`, every detector, threshold, tier, fold and evidence row. SN1 remains in WISHLIST as an idea until the trader moves it into `plan.md`; SN2, SN3 and SN4 landed in the 2026-09-12/13 sweep.

- **SN3 - one RRS pass (2026-09-12, WISHLIST sweep).** The cycle entered `run_rrs_scan` FOUR times - 5m, 15m, 1h, then again for whichever of the three the GUI had selected - and each entry walked the whole universe, re-bucketed the same 5-minute bars and rebuilt every symbol's `_build_intraday_rrs_profile` from scratch. That profile is an O(n^2) walk (one `real_relative_strength` per bar of the session over a growing slice) and does not depend on the timeframe at all, so three of the four builds were waste: 275 s of the 658 s preamble measured on 2026-09-08. It is now ONE walk. Everything timeframe-independent - the bars, the profile, the SPY context windows, the environment summary, the group-strength and internals snapshots - is measured once; aggregation, alignment and RRS run per timeframe inside the same walk against per-timeframe accumulators. `RRS_CYCLE_TIMEFRAME_KEYS` is `("5m", "15m", "1h")`, and a GUI selection outside that set (30m) is APPENDED to the same walk rather than given its own.
- **The seam is `rrs_payload_for`.** `BounceBot.rrs_payload_for(timeframe_key)` returns the payload this cycle produced for a timeframe, and `latest_rrs_payload` IS the entry for the GUI's selected timeframe - the same object, never a recomputed equal one - so exactly one `_decorate_snapshot` result reaches `bounce_service`'s `rrsSnapshotChanged` per cycle. `run_strategy` marks the stage once, `rrs_scan`, where it used to mark `rrs_scan_5m` / `_15m` / `_1h` / `_gui`.
- **The profile cache is keyed on the SYMBOL's last bar, not SPY's.** `_intraday_rrs_profile_for_cycle` caches on `(last bar dt, bar count, profile length, session date)`. SPY gaining a bar the symbol did not print cannot move the answer, because `_build_intraday_rrs_profile` aligns the symbol to SPY and drops the unmatched bar - so a name that printed nothing new is not re-profiled even when SPY moved. The bar count and the session date are in the key so a trimmed history or a day roll rebuilds when the last dt has not moved. Only today's rows are cached (<= 78 per symbol) and the cache is pruned to the scanned universe each cycle.
- **The numbers did not move, and that is pinned.** The 5m, 15m and 1h payloads are BYTE-IDENTICAL to what the four passes produced from the same bars: `tests/fixtures/ws_sn3_rrs_four_pass.json` was recorded on the pre-SN3 code at commit `204f4640` and is never regenerated (the provenance test asserts that commit literal). No formula, threshold, universe, ETF alignment or output field changed. `resolve_industry_ref_etf` is now handed the in-memory `industry_map_data` that `load_and_update_industry_etf_map` just wrote instead of re-reading the same JSON once per symbol per pass, and sector/industry reference ETF bars are bucketed once per (ETF, timeframe) per cycle instead of once per scanned symbol.
- **A scoring input is not something a pass-structure change may move.** `_record_environment_focus_history` is still called FOUR times per cycle in the same order (5m, 15m, 1h, GUI key), because its `hit_count` reaches the D1 attribute rows as `bouncebot.*_hit_count` (plan.md sec 5). What does change count is write-only diagnostics with no reader in the repo: `rrs_strength_scan.csv` gets one block per distinct timeframe instead of a duplicated fourth, `rrs_group_strength.csv` one block per cycle instead of four, and the industry map's `seen_count` advances once per symbol per cycle instead of four times.
- **Ask-first:** the trader's SN prompt is the yes for `run_rrs_scan`'s PASS STRUCTURE only. **Tests:** `tests/test_ws_sn3_one_rrs_pass.py` - five tests written red by the tester at `05adbefa` and proven to fail on the pre-change file.
- **What did not change:** `request_and_detect_bounce`, `_rebuild_feed`, every detector, threshold, tier, fold and evidence row, and every RRS number the passes produced. SN1 remains in WISHLIST as an idea until the trader moves it into `plan.md`; SN2 and SN4 landed in the same sweep.
- **File-scoped ask-first:** `bounce_bot_lib/legacy.py` is a detector file; the "Go" of 2026-09-08 is the yes for these two seams only.
- **Tests:** `tests/test_sn5_sn6_scanner_breath_and_fast_lane_order.py` - 12 of 13 fail with the fix reverted (proved on the lead's checkout, 2026-09-08).
- **SN4 - the feed diffs itself (2026-09-12, WISHLIST sweep).** A veto cost 4.0-4.1 s and one coalesced `focusChanged` 24.2 s on 2026-09-08 because both called `alert_center_panel._rebuild_feed`, which destroys and reconstructs up to MAX_FEED_ITEMS + MAX_D1_FEED_ITEMS = 350 row widget trees on the Qt thread. **`_feed_target_rows` is now the ONE statement of what the feed should look like and both paths read it**: `_sync_feed` reconciles the rows already on screen against it (destroy what is gone, insert what is missing, restyle what changed, and leave every other row as the SAME widget at the same position), and `_rebuild_feed` builds every row from it. A veto and the coalesced Focus refresh call the diff; the rebuild is kept for the three whole-feed decisions - the minimum-tier switch, the day's Clear, and `_unpin_d1_focus`. Measured offscreen on 250 M5 + 100 D1 rows: **veto 8.5 ms (was 220.1 ms), coalesced focus flush 6.2 ms (was 223.7 ms)**, and neither constructs a row widget.
- **The two defects the parity exposed.** `_rebuild_feed` was not fold-aware - a name that has alerted three times has three entries in `self._alerts` and ONE row, and the rebuild drew three - and it dropped every ×N badge because it passed no `repeat`. So the target keeps **one row per (symbol, side) at the OLDEST qualifying entry's position** (where the fold left it) and re-stamps the count from `RepetitionLedger.repeat_counts()`, a read-only snapshot: `consider` is a DECISION and calling it again for a redraw would count the alert twice. A name that ESCALATED therefore collapses from its momentary two rows (the re-floated one plus the original) to one row at the original position on the next refresh.
- **The digest is a registry, not a side effect.** `_digested_keys` records the (symbol, side) keys the open-burst row is standing in for, day-scoped with the ledger, and `_refresh_open_digest_row` creates, updates or retires the row from it. Before this a veto during the burst destroyed the digest row and drew a row per digested alert - the pile-up the digest exists to prevent. Nothing is withheld anywhere in this chain: the backing lists, the ledger, the review queue, History and every evidence stream are written before any of it.
- **The Focus half is one widget's worth of work.** `AlertFeedItem.apply_focus_state` sets the star's `focusOn` property, the gold `alertKind` frame and the ★ SWING / ★ M5 badge, unpolishes and polishes THAT widget, and no-ops when the category is unchanged - so a like touches the rows it names and nothing else. No stylesheet is re-set on the panel.
- **File-scoped ask-first (SN4):** `ui/panels/alert_center_panel.py` is an alert file; the SN prompt is the yes for `_rebuild_feed` / `_insert_item_into` / `_ignore_alert_symbol` and the feed's row bookkeeping ONLY. No alert gate, tier, threshold or evidence row changed.
- **Tests (SN4):** `tests/test_ws_sn4_feed_diff.py` (the packet's, 5 of 8 red before the fix) and `tests/test_ws_sn4_feed_diff_builder.py` (4 of 5 red before it).

- **SN2 - the M5 window is fetched whole once a day, then extended (2026-09-13, WISHLIST sweep).** `request_and_detect_bounce` asked IB for `durationStr="5 D"` of 5-minute bars for EVERY symbol on EVERY cycle: ~390 bars and ~206 KB a symbol, 586 symbol scans a cycle, ~120 MB off the wire and ~230,000 rows appended one at a time by the `historicalData` callback every 25 minutes - all of it on the interpreter lock the GUI needs. The window is now fetched WHOLE once per symbol per market-local day and KEPT (`_sn2_bar_windows`, the raw row dicts the callback appended); every later cycle asks only for the bars since the last completed one, in seconds, and merges them onto the kept rows. **The frame the detectors read is the frame a fresh "5 D" fetch would have produced** - same rows, same order, same dtypes, same RangeIndex - which is the whole promise and what `tests/fixtures/ws_sn2_fresh_frames.json` pins. Nothing below the fetch changed: `_dedupe_bars(_bars_to_ib(...))`, `pd.DataFrame(all_bars)`, every detector, threshold, tier, fold and evidence row are untouched.
- **The delta carries its own proof.** It reaches one bar further back than the gap (`SN2_DELTA_MARGIN_SECONDS`, 300 s), so the kept window's last bar comes back WITH it; `_sn2_same_bar` requires the overlap bar's dt AND close to match, and a mismatch throws the window away and refetches "5 D" rather than splicing two different series together - a revision, a corporate action, a halt. A delta that never reached the kept window's last bar, a delta carrying a session the kept window does not have (the pre-market first cycle, where a later "5 D" would have dropped the oldest session), a market-local day roll, a gap of a day or more, and rows that are unreadable or out of order all refetch whole. The ten-bar short-data guard moved to the MERGED result, because a 25-minute cycle's delta is about five bars.
- **A forming bar never enters the kept window, and that is a DEVIATION from the packet.** The packet and the tester asked for a forming tail to force a whole-window refetch. IB does not make that affordable: an `endDateTime=""` request's last row is ALWAYS the bar still forming (`_rows_after_bounce_entry_for_session` in this same file says so, and `_m5_bar_completed` exists because "IB's historical cache has no complete/forming marker"), so that rule refetches the whole window on every cycle of a live session and SN2 saves nothing - it cannot hold at the same time as the packet's own "one delta request per cycle". So `_sn2_keep_window` keeps only the bars that had CLOSED when they were served (`completed_bars.is_completed_bar`, the one rule) and the next delta re-reads the forming one once it has closed. The invariant plan.md sec 5 states is kept strictly: a preview price is never written into a later frame as if it were final. The packet's mechanism is still there and still proven, one constant away - `BounceBot.SN2_FORMING_TAIL_FORCES_REFETCH`, False by default - so the trader's call is a one-line change. `tests/test_ws_sn2_incremental_bars.py::test_a_forming_last_bar_forces_a_whole_window_refetch_and_the_cache_recovers` is RED by that decision and was left red rather than weakened.
- **The cache is bounded by the scanned set.** `_prune_latest_bars_for_cycle` is the one cycle-start hook and now frees the SN2 window of every symbol the cycle no longer scans (and resets the per-cycle counters); the call site names the scanned set (`scanned_symbols=all_symbols`) so a priority symbol's window survives a background-refresh cycle, and a caller that does not name it is taken to have named the keep set in `background_symbols` - which is how the packet's test drives it. The hook is also called UNBOUND on a bare stub by `tests/test_bounce_learning.py`, so the SN2 calls are looked up before they are made. ~600 symbols x ~390 rows of dicts is roughly 90-100 MB held steady, in place of the same amount churned every cycle.
- **The measurement is one log line per cycle:** `SN2 M5 window fetch, cycle N: A full window(s) = B bar(s), C delta(s) = D bar(s); E bar(s) fetched in total` (`_sn2_log_cycle_fetch`, emitted after the sweep). After the first cycle of a session the full-window count should be the handful of names that are new or that fell out of line, not the scanned set.
- **Ask-first (SN2):** the trader's SN prompt is the yes for `request_and_detect_bounce`'s FETCH AND CACHE only. **Tests:** `tests/test_ws_sn2_incremental_bars.py` (the tester's, 6 of 8 red before the fix, 1 left red by the deviation above) and `tests/test_ws_sn2_incremental_bars_builder.py` (5 tests, all 5 red on the restored pre-change file).

---

## WS-PT4 - the AWAY digest ranks by points when the trader's switch is on (2026-09-12)

**What the trader said** (WISHLIST item 4, points, block 4): *"rank the AWAY digest's
swing picks by points too (today it ranks by the family's Wilson bound)"* - marked "open,
your call". The lead's ruling, which the trader may overrule: the digest uses the EXISTING
Points switch and nothing new. `setup_points.rank_enabled()` (local setting
`rank_setups_by_points`, default OFF) is read AT SORT TIME inside
`autopilot_core.order_swing_picks`; OFF is the identity function and the digest keeps the
Wilson order exactly, which is what the golden `tests/fixtures/ws_pt4_away_digest_switch_off.txt`
pins. ON, `setup_points.rank_order` puts the `RANKED_BUCKETS` rows (favourite,
near-favourite, high-conviction) first by total and every other pick after them; the list
handed to it is already in the Wilson order, so that order is the tiebreak and the order
the unranked rows keep - the same shape as the setups table, where the point ranking is
applied after the Working-lately order. The near cap is still applied AFTER the ranking by
the renderer, so what a cap hides is the weakest near row by whichever order is in force
and never the best one; the bucket is still printed and never ranked on.

**One scorer, two callers.** The digest scores with `setup_points.score_row` itself -
there is no second formula - so `autopilot_core.swing_pick_projection` widens the digest's
pick row to carry the SCAN ROW (`raw`) and the two group-context readings
(`d1_vs_sector`, `d1_vs_industry`) the display enrichment already attached, plus
`bucket_key`: the phone prints the bucket LABEL ("Favorite") and `RANKED_BUCKETS` matches
the KEY (`favorite_setup`), so a ranking that matched the label would quietly rank nothing.
The family record is `swing_family_points_record`, which turns the digest's own
`swing_family_read()` counts into the `win_rate_lb` shape the desk panel injects - the
same `swing_evidence.read_eligible_rows` under `POLICY_SCANROW_V1` in the same lately
window, the same `swing_headline.wilson_lower_bound`, so the bound the digest scores on is
the bound it used to order on and the one the setups table shows. Nothing is re-derived
from a different source. A pick missing an input scores that part 0 with the note and is
never dropped (an ungraded family scores 0 on `setup` and says so), and any failure in the
whole path falls back to the Wilson order rather than costing the swing block.

**The digest says which order it used.** The `Ranked on:` line now ends in
`| order: Wilson bound` or `| order: points (switch on)`, and is written whenever picks
were ranked even if the record line is missing - a points ranking that never says so is
the defect ST1 item 3 fixed for the Wilson one. Presentation only: nothing here reaches a
detector, a score, an alert, a watchlist, Focus or `review_policy.json`, no weight is
tuned (`setup_points.active_weights()` is read, never written), and the phone PUSH
(`build_swing_push`) is deliberately untouched - it is a different, shorter list.

**Tests:** `tests/test_ws_pt4_digest_points.py` - 10 of 11 fail with
`scripts/autopilot_core.py` and `scripts/ui/services/autopilot_service.py` restored
(proved 2026-09-12); the eleventh asserts the switch-OFF near cap, which must pass both ways.

---

## Workspace memory is recall, never authority (2026-09-12, WISHLIST 11)

**The trader, verbatim:** *"we will begin integrating wishlist.md this week. for now
integrate the memory changes then standby"* (2026-09-12). The item itself was added on
2026-09-11 when the trader asked to bring the memory changes from their JumpStarter repo
into this wishlist.

**The source.** `Isidore94/JumpStarter`, `main` at `664e083` (2026-09-11): commits
`48d86e9` (hierarchical memory), `3234bd2` (index authority), `556cffd` (maintenance and
role guidance), `664e083` (integration and Codex verification). Read there: `CLAUDE.md`
"Workspace memory", `MEMORY.md`, `memory/`, `docs/INTERNALS.md` "Workspace memory is
request-grounded", `docs/CODEX_NOTES.md` and the `.codex/agents/*.toml` paragraphs. M1
changed JumpStarter's own workspace guidance and not its templates, so `jumpstart init`
installs none of it; everything here was adapted by hand and the clean local checkout at
`C:\Users\Aaron\JumpStarter` matched GitHub at that revision.

**What carried over unchanged.** A root `MEMORY.md` that routes (name, file, trigger
keywords) and states no fact; detail under `memory/` split into `people/`, `projects/`,
`decisions/`, dated daily notes and a prunable `context/`; "search memory first" before
answering about prior work, decisions, dates, people or preferences, at most five sources
for that answer, every fact cited by file, tag and date; the four provenance tags with a
date and a source on every non-blank detail line; supersession in place (strike the old
line with its date, the replacement beside it); the index updated in the same commit as
the detail; the weighted lesson threshold (three independent signals across two sessions,
a signal older than 30 days counting half, a trader correction applying at once, a failure
memory describing and never instructing); the 15,000-character file convention; and the
role split (recon and reviewer propose, tester proposes while writing red tests, a builder
records only an in-scope durable detail, the lead integrates).

**What this repo changed, and why.**

- *Authority.* JumpStarter says "detail files are authoritative". Here that clause is
  scoped to detail versus its own index. `CURRENT_CHECKPOINT.md` stays the brief,
  `plan.md` the build order and promotion authority, `CHANGELOG.md` the inventory, the
  decision records and specs the contracts, and the code the fact. A memory line that
  disagrees with any of those is the defect. The reason is the repo's own history: every
  rule in `CLAUDE.md` exists because something broke, and a recalled preference must never
  outrank a measured one.
- *The lesson threshold authorizes nothing.* Three signals across two sessions may promote
  a line from `[inferred]` to a standing lesson inside `memory/`. They never authorize a
  detector, score or alert change, never overwrite an accepted decision, never promote a
  WISHLIST item and never bypass the file-scoped ask-first rule. Only the trader does those.
- *Idle boot never weakens the narrow reads.* JumpStarter reads only identity plus
  `MEMORY.md` at idle boot. So does this repo, but the moment a task exists the mandatory
  workflow (glance block, plan sections 5-7, the CHANGELOG inventory search, `docs/README.md`)
  applies exactly as before; memory adds a read, it removes none.
- *A live-status question is never answered from memory.* "Where are we", "is the desk
  running", "which branch" go to the checkpoint and the code. Memory lines carry dates for
  this reason; an undated or stale line is unknown, not true.
- *Claude's auto-memory stays machine-local.* Claude Code keeps a private memory folder
  under the Claude project directory on this machine. It was inventoried (28 files on
  2026-09-12) and only the trader's standing statements and the non-re-derivable lessons
  were seeded into `memory/`; dated audit and assessment notes stayed there because the
  checkpoint archives already hold them, and broker or tax facts are never written into
  the repository. That folder is scratch for one tool on one machine; `memory/` is the
  record both tools share.
- *No sync tool.* JumpStarter verifies `AGENTS.md` against `CLAUDE.md` by sha256 through
  `jumpstart check`. This repo copies by hand and proves it with `cmp`; nothing from
  JumpStarter's CLI, operator facts, decision 0002 or approval record was imported.

**Verification.** Static, on the commit: every `MEMORY.md` route resolves to a file or
directory; every non-blank line in `people/`, `projects/` and `decisions/` carries a tag,
a date and a source; `CLAUDE.md` and `AGENTS.md` are byte-identical; the eight role files
carry the paragraph. Owed as gate #93: in a fresh Claude session and a fresh Codex session,
a bounded prior-preference question reads only the matching detail file and cites file,
tag and date; a live-status question reads the checkpoint and the code; a recon or
reviewer run writes nothing under `memory/`; nothing recalled is offered as authorization
for app work.

**Reopen trigger.** The trader changes the recall policy, the tags, the caps or the scope
of what memory may hold; or the root instruction-file trim moves this section.

## 5D - a watchlist edit is a dated event, never a verdict (2026-09-12, WISHLIST sweep)

**The gap.** The four plain watchlists - `longs.txt`, `shorts.txt`, `swinglongs.txt`,
`shortswings.txt` - are the oldest surface on the desk and the only trader act that kept
no history at all. `WatchlistEditorPanel._write_symbols` wrote the joined symbols to the
file and that was the entire record: a name appeared, a name vanished, and nothing said
who did it or when. Every other verdict already has a forward record (P5: veto, like,
pass, rejection), so the act the trader performs most often was the one the evidence loop
could not see. `focus_membership_events` covers Focus-pick episodes, which is a different
membership - hence a new stream, in the same shape, not a new schema style.

**The rule.** `scripts/watchlist_intent_events.py` (schema `watchlist_intent_event_v1`,
stream `WATCHLIST_INTENT_EVENTS_FILE` in the shared home) appends one JSONL row per symbol
that joined or left one of the four lists: `ts` (aware, market-local, the OBSERVATION
time), `market_date`, `list`, `side`, `horizon`, `symbol`, `action`, `source`, `reason`
and `writer`. Five clauses bind it:

- **Membership is interest, never a claim.** Not a setup, not a position, not a
  prediction. A `remove` is not a dislike - the dislike has its own store
  (`pick_feedback.jsonl`) and the trader's own words.
- **Nothing is invented.** `ts` is when the desk SAW the change. An edit made in Notepad
  or on the DAS while the app was closed is stamped at the moment the panel next loaded
  the file and labelled `observed_external`; it is never back-dated to a time nobody
  measured, and it is never asserted to be a trader decision.
- **A machine write is distinguishable.** `FocusPickStore._inject_into_shared` /
  `_uninject_from_shared` write `machine_inject` / `machine_uninject` through the same
  writer; the panel writes `trader_edit`, and a clipboard drop `trader_paste`. The
  cross-side removal (`WatchlistEditorArea._handle_symbols_saved`) is the trader's edit
  with its cause in `reason`, because the trader caused it.
- **The evidence never costs the save.** `_write_symbols` writes the FILE first, then
  appends; every writer returns a count and swallows its own failure; a failed append
  shows `(intent not recorded)` as a status suffix and nothing more. Hand-entered names
  are untouched by every path (plan.md sec 5).
- **Re-ordering is not a change.** The diff is against what the panel last read or wrote,
  so `sort_symbols` and an autosave that moved nothing append nothing, and a re-add after
  a removal is a new `add`.

**The baseline row, and why it stays small.** Reconstructing membership needs a starting
point, and the file's current contents are not one - they are today's state, not the state
when the stream began. So the first time a list is seen with no reconstructable history,
ONE `baseline_recorded` row names the symbols then present and NO `add` rows are written,
because nobody observed those additions. It is written at most once per list (never per
load), encodes the symbols as one comma-joined string rather than a list of objects, and
carries `symbol_count` and a 12-character `symbols_digest`. After that the stream grows
with CHANGES, never with loads: three reopenings of the page write nothing.

**What reads it.** Nothing, yet. No detector, score, alert, Focus list, scanner or
`review_policy.json` touches it. 10G's Watchlist tab will render source badges from it and
the WS-DR work may join on it; until then `python -m watchlist_intent_events tail --list
longs` (run from `scripts/`) is the whole consumer.

**Known gap.** `autopilot_core`'s auto-populate (`write_bouncebot_watchlists` and the
append helper near `autopilot_core.py:3225`) is a THIRD machine writer of `longs.txt` /
`shorts.txt` and is not labelled: its adds surface as `observed_external` at the next
panel load. That is honest - the desk did observe them late - but coarse, and labelling
that seam is the obvious follow-on. The packet named only the Focus store's two seams.

**Reopen trigger.** A second reader appears (10G, WS-DR), the auto-populate seam is
labelled, or the trader asks to be prompted for a reason on an edit - which this packet
deliberately does not do.

## FC1 - a forming bar never reaches the daily-bar cache (2026-09-12, WISHLIST sweep)

### What was measured, read-only on the live machine cache

`%LOCALAPPDATA%\TradingBotV3\machine_cache\daily_bars` held **1,988 per-symbol CSVs, 66 of
which ended in a candle that could not have happened**: the last row's open sits outside
its own `[low, high]`. `ADC 2026-09-11 O=71.870 H=71.805 L=70.970 C=71.230` is the shape.
Always exactly one bad row, always the last. The last-row dates cluster on the sessions the
desk scanned while they were still open - 2026-09-11 x55, 2026-09-10 x4, 2026-09-04 x2, one
each on 2026-09-02, 2026-09-01, 2026-07-07, 2026-06-08, 2026-05-15 - which is the signature
of a FORMING session bar: Yahoo returns today's partial bar during the session, the high had
not yet grown to contain the open, the scan wrote it, and no later refresh replaced it.

The count moved with the calendar (the trader's own 2026-09-06 read was 100 of 1,980 on the
2026-09-04 cluster, this one 66 of 1,988 on the 2026-09-11 cluster), which is itself the
finding: this was not one bad afternoon, it was every scan that ran before the close.

The writer seam was `master_avwap_lib.legacy._write_cached_daily_bar_frame`, reached from
`fetch_daily_bars` after `_merge_daily_bar_frames`. Neither the merge nor the write consulted
`scripts/completed_bars.py` or the candle invariant, so whatever the provider returned at
11:00 is what landed on disk. `_seed_daily_bar_cache_from_durable` is the second writer of
the same CSV and had the same gap.

### The rules this produced

- **A daily bar may be stored only once the exchange session for its date has closed**, and
  the comparison is made in exchange time with `astimezone`, never `replace(tzinfo=None)`.
  That is `completed_bars.py`'s rule stated for a session-length bar.
- **`market_calendar` models no early close**, deliberately (its own docstring), so every
  session is judged against 16:00 ET. That is conservative in the only direction that
  matters: a half-day's bar is called complete at 16:00 rather than at 13:00, so a forming
  bar is never called finished. A 13:00-close session's bar simply waits three hours.
- **A stored candle must be possible** (`low <= open, close <= high`). Completion alone does
  not make a candle real: MCW 2026-06-08 and TERN 2026-05-15 are months old and still
  impossible.
- **A row that is both forming and impossible counts ONCE, as forming**, because forming is
  the cause. That is what makes `kept + forming_dropped + invalid_dropped == fetched` hold
  exactly, and the run manifest publishes both counts under `daily_bars_forming_dropped` and
  `daily_bars_invalid_dropped` with one INFO line per scan naming both - **even at zero**,
  because a counter that only appears on a bad day cannot be checked on a good one.
- **The rule, the counters and the repair live in `scripts/master_avwap_lib/daily_bar_cache.py`,
  not in `legacy.py`.** The trader's FC1 prompt is the ask-first yes for the cache WRITER
  seam only, so the ask-first file gained an import and two call sites and nothing else.
- **When every offered row is refused, the cache file is left alone** rather than replaced by
  an empty one. Missing data is uncertainty, never confirmation.
- **The seed filters the FILE and not the answer.** `_seed_daily_bar_cache_from_durable`
  writes the filtered frame to the CSV and hands the caller the durable store's own frame
  unchanged, so no reader sees a different series because of this packet.

### The repair, and what the bad rows touched

`cd scripts && python -m master_avwap_lib.daily_bar_cache repair [--apply]` - dry run by
default. It prints `project_paths.DATA_DIR` and the cache directory before it reads anything,
refuses a target under `C:\TradingBotData`, and for a file whose LAST row is forming or
impossible refetches that session through the pinned Yahoo path and writes temp-and-rename.
**Only the last row is ever touched**: an interior oddity is a data question this tool does
not get to answer. The refetch window is widened to REACH the bad session - a fixed ten-day
window answered "the provider has no bar for that session" for PRKS 2026-07-07, which
measures the request and not Yahoo. A session that is still open replaces nothing and says so.

Dry run on a COPY of the live cache (2026-09-12): 1,988 files read, 66 repairable, 64 with a
completed replacement bar; MCW and TERN return no data from Yahoo at all, so their rows would
be removed with no replacement. On a COPY of the tracker's SQLite mirror, 625 records carry
one of the 66 symbols and **443 have the bad date inside their replay window** (270 setups,
169 studies, 4 controls): 443 had AVWAP band levels recomputed from that bar, 441 were marked
to market on it, and **zero had a fill booked on it** - `gap_aware_v2` already refuses an
invalid bar (`skipped_bar_reasons: invalid_bar`, 4,660 times across those scenarios). The next
persisted tracker write rebuilds every record, so no tracker repair is owed.

### Reopen trigger

`market_calendar` gains an early-close model (then the 16:00 ET judgement becomes the
session's real close); the daily source stops being Yahoo; or a second writer of the
per-symbol CSV appears. The durable Parquet mirror (`_persist_durable_daily_bars`) still
receives the unfiltered merged frame - it is outside the FC1 yes and is an open question for
the trader, not a silent edit.

---

## SX - the star and the X are the day's decisions (2026-09-12, WISHLIST item 8)

**The trader, verbatim (2026-09-10, WISHLIST item 8):** the ★ is filled for anything in
Focus **and** anything liked today, and the ✕ is painted **bright red** when the trader
has "vetoed, disliked, passed or skipped that symbol today", with the tooltip saying
which decision and when. The lead's answer to the one open question (2026-09-12): a name
both liked and vetoed today shows **both** marks - they are two independent facts and
neither cancels the other.

**Presentation only.** Nothing is hidden, re-ordered, muted or written by this. The
movers-only filter, the Points switch, the bucket filter and the sort are untouched, and
no row moves because a decision was made about it.

**The wider read, from the same parse.** `pick_feedback.decisions_today()` answers a
WIDER question than its sibling `reviewed_symbols_today()`: the day-trade PASS annotation
and the M5 click-away (`action: "skip"` with `detail.reason ==
"clicked_away_from_m5_alert"`, written at `alert_center_panel.py`'s skip branch - a click
away IS a pass, trader 2026-09-01) are decisions the trader made and were never in the
"Reviewed today" badge's filters. Widening that badge would have moved a shipped surface,
so both answers now come out of ONE cached, mtime-keyed parse (`_read_day_ledgers`), and
`reviewed_symbols_today`'s returned set is what it always was. `unfavorite` is in NEITHER
map: taking a name out of Focus is not a verdict on it (P5).

**The kinds are fixed, because the tooltip is built from them.** Liked: `quick`,
`claimed` (a `like_claim` row with no `like_mode` key at all reads `claimed` - a claim was
required until P9), `like` (a `pick_feedback` star). Rejected: `veto`, `dislike`,
`not_today`, `pass`, `m5_click_away`, `remove_today`. A bare rail `skip` ("skip for now")
is not one of them and is not modelled here.

**The colour is a token, in both themes.** `theme.color()` answers an unknown name with
`neutral`, so a `reject_today` present in only one `THEMES` dict would paint the mark grey
in the other and nothing would raise. It is deliberately not `short`: that one is a SIDE
and has to sit calmly beside `long` in every chip and score bar.

**Never a file read in `paint`.** The delegate holds a lookup over one already-parsed
snapshot (`SetupTableDelegate.set_decision_lookup`, per-symbol views memoized inside
`DayDecisions`); the panel rebuilds it on a `_DayDecisionsWorker` QThread and repaints
only when the payload actually changed. Every trigger - a capture verb, a scan's
`set_rows`, `showEvent`, the day roll checked on the 30 s report poll - goes through the
SAME `SignalCoalescer` a Focus change uses, so three verbs in one event-loop slot are one
repaint (the 2026-08-31 rule; that viewport pass was the hottest stack in the stall log).

**Tests:** `tests/test_ws_sx_star_x.py` - 14 red on the pre-fix branch tip, all green
after. The marks are asserted by rendering the two cells offscreen and sampling the
PIXELS for the exact token, because "bright red" is a claim about what the trader sees.

**Reopen trigger.** A new capture verb joins the decision family; the trader asks for a
third mark or for one of the two to mean something else.

## AI1 - an enrichment row is never blank on success (2026-09-12, WISHLIST sweep)

**The trader, verbatim** (WISHLIST 10K, "Fable's integration sequence", step 1):
*"Repair trustworthy output. Reproduce enrichment failure before fixing its
schema/extractors, refusal of empty success and retry/supersession of existing blank
records. Verify a real saved suggestion reaches the trader. Distinguish published,
synthesized, useful-empty/abstained, failed and partial outputs. ... do not bypass gates,
raise timeouts blindly or change model."*

**What was measured, code against code, 2026-09-12.** `ai_jobs/enrichment.py`'s
`_proposed_tags` read `tags` / `setups` / `families` and `_summary_text` read `headline` /
`summary` / `what_worked` / `lessons`. The response it validated was
`ai_summary.AI_SUMMARY_JSON_SCHEMA`: `additionalProperties: False` over
`executive_summary` plus `what_is_working`, `what_is_not_working`, `best_candidates`,
`lessons_for_tomorrow`, `risk_notes`. **Not one of the seven keys the two extractors read
could survive that validation.** Six distinct trades over 2026-09-09..11 therefore carried
a blank `ai_trade_enrichment` row while the `journal_enrichment` ledger row said `ok`, and
`_trades_for_session` skipped a trade when ANY row existed for the session - so the blank
satisfied "already done" permanently, and fixing the schema alone would still not have
produced a single non-blank row.

Two nights earlier the same shape of defect had eaten the main summary. `briefs.py`
returned `STATUS_OK` for the chunked branch whatever the reason said, and `map_reduce`
signalled a lost synthesis only as `map_reduce.synthesized`, a boolean two levels inside
the result that `briefs.py` did not read. The 900 s synthesis timeouts of 2026-09-10 and
-11 published an unsynthesized fallback and were ledgered as clean nights.

**Three failures, one cause.** In each case a document said one thing and its reader
assumed another, and nothing in between ever compared the two. The fixes are all the same
move: put the fact in the payload as a WORD, and make the reader read it.

**The rules this produced.**

1. **One provider path, two contracts.** `ENRICHMENT_JSON_SCHEMA` (`summary`, `tags`,
   `confidence`, `sources`, `unknowns`, closed) is the per-trade contract;
   `request_ai_summary` gained `schema` / `schema_name` / `prompt_version` with the
   session summary's schema as the default, so every existing caller's request payload is
   byte-identical. A second provider function was rejected on sight: it would be a second
   place for the timeout, the retry, the truncation tripwire and the length-stop rule to
   drift. `_proposed_tags` / `_summary_text` remain the ONE extraction seam and read this
   schema's keys and no others - the old fallback chains looked tolerant and were three
   chances at a key the contract forbade.
2. **An empty answer is an ANSWER.** `status` is `enriched` / `abstained` / `failed`,
   `reason` is the model's own `unknowns` or the error class, and the slot reports
   `enriched A, abstained B, failed C of N` with `STATUS_OK` only for `A + B == N, C == 0`.
   A provider failure is now RECORDED against the trade it happened to; before, nothing
   was written and the outage lived in one log sentence.
3. **A blank row is not a finished trade.** The legacy blank is blank summary AND blank
   tags AND no status - all three, because an abstention is blank in the first two and is
   real. The repair APPENDS a row naming the one it replaces (`supersedes_row_id`);
   nothing is rewritten, so the history of a repaired night stays readable. A settled row
   ends the trade's attempt for the session; a `failed` row does not, so a second firing
   in the window retries rather than reporting "nothing to enrich".
4. **Every published summary names its completion.** `synthesized` / `partial` /
   `unsynthesized_fallback` / `failed`, top level, always present; a lost synthesis
   outranks a lost slice. `STATUS_OK` only for `synthesized`; the document is still
   published, because losing the findings would be worse than publishing them
   unsynthesized. **The timeout was not raised and the model was not changed.**
5. **The advice has a reader.** `journal_feed.latest_ai_enrichment` (newest row nothing
   supersedes) rendered by `TradesTab._show_trade`, marked advisory, with status,
   confidence and `written_at`. An `abstained` or `failed` row is SHOWN. A table nobody
   could open was indistinguishable from a table nobody wrote to.
6. **The preference section is selected by size, never by result.** `preference_to_trade`
   in the nightly package: three grains kept apart, coverage derived from the report's own
   `match_basis` plus the 10-SESSION window, 20 examples by `(session_date, row order)`
   descending. `journal_r` may be READ in an example and may never RANK one - the live
   report's single best row is also its oldest, and a section that surfaced it would be
   teaching the model that the trader's stated preferences work better than they do.
   It is ON the nightly slate (lead decision 2026-09-12, the trader able to overrule:
   "into the existing AI package" means the package that actually reaches the trader),
   and it earns its place by being small - **7,668 chars measured on a read-only copy of
   the live report (838 statement rows, 136,720 bytes on disk, 2026-09-11)**: 48% of the
   16,000-char per-source cap, 9.6% of the 80,000-char package budget, a 17.8x reduction
   that does not decay as the report grows because the counts are fixed-size and the
   examples are capped. Budget weight 2 costs the other five scopes nothing, because the
   allocator caps a scope at what it needs and returns the surplus; weight 1's base share
   of 6,666 would simply have left it depending on a surplus paid to heavier scopes first.

**Two defects found beside it, both in the completion-word path.**
`operations_audit._ai_jobs_check` counted `statuses.get("degraded")` while the ledger's
constant is `degraded_no_narrative`, so no AI job has EVER been able to show as degraded on
the System Health strip; and it read `ts` / `timestamp` while `ledger.record` writes
`started_at` / `finished_at`, so every AI row read as undated and the freshness branch
could never fire. Both fixed here because both stood between the new word and the trader's
eye.

**Reopen trigger.** The enrichment schema gains or loses a field; the completion
vocabulary changes; a second provider path is proposed; or the preference section's
selection key is asked to consider a result.

## ENV - one D1 environment label per session, joined by scan date (2026-09-12, WISHLIST 7)

**What the trader asked for.** WISHLIST item 7: read the readouts cut by the kind of day
the market was having. The only honest way to do that is to decide the kind of day ONCE,
from completed daily bars, under a named and versioned rule - not per surface, per reader,
or per memory of what February felt like. Lead rulings for the sweep (the trader may
overrule): the `legacy.py` stamp on the tracker outcome row is NOT built here (ask-first;
the dated store joined on `scan_date` gives the same point-in-time answer); the rule is
`d1_environment_v1` below; and the label only LABELS a readout - no weight is set per
environment for the point system.

**The rule, in one pure module.** `scripts/indicators/d1_environment.py`
(`classify_environment`), bars in and a frozen `D1Environment` out, `None` for anything
unmeasurable, no clock, no I/O, no engine import. In this ORDER:

    len(bars) < WARMUP_SESSIONS (34)                -> unknown, reason "warmup"
    atr14 unmeasurable                              -> unknown, reason "unmeasurable"
    range_atr <= COMPRESSION_RANGE_ATR (3.0)        -> compressed
    slope_atr > +TREND_SLOPE_ATR (0.5), close>sma20 -> trending_up
    slope_atr < -TREND_SLOPE_ATR,       close<sma20 -> trending_down
    otherwise                                       -> mixed

`range_atr` is (max high - min low over the last 10 sessions) / ATR14 and `slope_atr` is
(SMA20 today - SMA20 ten sessions ago) / ATR14. **ATR14 is Wilder at the LAST bar over the
WHOLE supplied series**, the 10-session window included - the packet left that open and
this is the answer; `indicators.atr.wilder_atr` owns the recurrence, so there is no fourth
copy of Wilder's smoothing on the desk. The warm-up is 34 because SMA20 ten sessions back
needs 30 bars and ATR14 needs 15.

**Compression is decided FIRST, deliberately.** A quiet market grinding higher inside a
three-ATR box is a compressed market; calling it a trend is how a reader talks themself
into size on a day that never went anywhere. On the recorded SPY series 2026-04-29 is
exactly that: `slope_atr` 3.84 with the close above its SMA20, `range_atr` 2.19, labelled
`compressed`. 2026-09-11 lands at `range_atr` 3.0030 - three thousandths the wrong side of
the threshold - which is why the golden pins six sessions to the last bit.

**The store.** `scripts/d1_environment_store.py` appends one JSONL row per `(session,
benchmark, rule_version)` to `project_paths.D1_ENVIRONMENT_FILE` in the shared home. A key
already on disk is NEVER rewritten (`append_environment` returns False and touches
nothing): a re-run or a repaired bar file is a second opinion about a day that already has
one. A new rule is a new VERSION beside the old one, and that is what lets both live in one
file - `label_for_session` is asked for a version and answers only from its rows. Each
benchmark keeps its own row (SPY / QQQ / IWM), never pooled and never averaged. `unknown`
is an ANSWER and is written with its reason; a missing row and a measured `unknown` both
read `unknown`, and only the row's `reason` tells them apart. `labels_by_session` is ONE
read cached by the file's MTIME (cheap enough for the Results worker); `read_rows` is
deliberately uncached because the writer asks it for the keys it must not duplicate.

**The hook.** `runner.record_d1_environment` runs as a SIBLING of
`bridge_earnings_anchor_caches_to_csv` at the end of a scan - after the caches are saved,
before `save_history`, one call site. It fetches SPY/QQQ/IWM through the SAME pinned
`fetch_daily_bars` the scan itself uses (never a second provider path), drops the FORMING
bar through `completed_bars.is_completed_bar` at daily length, calls the pure rule and
appends. It returns `{benchmark: label}` for the log line only; every failure is logged and
swallowed, because an evidence store never costs the thing it records. The log line is
`D1 environment: SPY=<label> QQQ=.. IWM=.. (d1_environment_v1, bars through <session>)`.

**The join is by SCAN DATE and never the exit date.**
`d1_environment_join.attach_environment` adds `d1_environment` to each row IN PLACE (the
caller's own list and dicts - the Results worker joins tens of thousands of rows on a
redraw). Both dates are usually in the store, so joining on the wrong column is not a
blank; it is a plausible label pointing the wrong way. A present-and-empty date, a missing
column and an unlabelled session all read `unknown`.

**The readout.** Research > Results gains "By environment (SPY, d1_environment_v1)" under
Bot x Swing only - a My-trades page has no scan date to join on. One row per (environment,
side): win rate FIRST with `n` and the ONE Wilson bound from `swing_headline`, SORTED BY
THE BOUND (62% on a hundred above 67% on thirty), the `MIN_REPORTABLE_N` floor LABELLING a
row and never hiding it, and `unknown` as its own row pooled into nothing. The rows are
`favorable_direction` (ST1), so the columns are headed "Favorable %" and the unit is `%`.
The section carries NO verdict line - it names no leader - and the panel now skips a
section with no verdict rather than printing a blank one above the cards. The champion
sections are byte-identical with and without the cut.

**Why the cut reads 120 sessions and not "lately".** `ENVIRONMENT_WINDOW_SESSIONS` is
`6 * LATELY_SESSIONS`. Twenty sessions of SPY is usually ONE environment, so a cut of
"lately" would print one populated row and four empty ones and answer nothing. It is a
declared parameter of THIS readout, read by nothing else, and every row's sentence says the
number out loud. It is not a second definition of "lately".

**The backfill.** `python -m d1_environment_store backfill --benchmark SPY --since
2026-01-01` (from `scripts/`) is DRY BY DEFAULT, prints `DATA_DIR`, the store and the bar
cache BEFORE anything (the 2026-09-05 rule), labels each past session POINT-IN-TIME under
the current version with `source = backfill`, and never relabels a session already written.
Run DRY on 2026-09-12 against a COPY of the machine cache, 184 sessions 2025-12-17 ..
2026-09-11: **SPY** 69 compressed / 43 mixed / 33 unknown (the warm-up) / 27 trending_up /
12 trending_down; **QQQ** 64 / 40 / 33 / 32 / 15; **IWM** 84 / 27 / 33 / 32 / 8. `--apply`
on the live store is the trader's call.

**Shadow only.** Nothing in this chain reaches a detector, score, alert, watchlist, Focus
list, review queue or `review_policy.json` (plan.md sec 5), and no `legacy.py` line was
edited for it.

**Reopen trigger.** The trader wants the label stamped on the tracker outcome row itself
(an ask-first `legacy.py` change), wants a weight per environment in the point system, or
wants a per-family cut inside the section - none of which this packet does.

## TH - the theta picks are graded, never changed (2026-09-12, WISHLIST sweep)

WISHLIST item 6, in the trader's words: *"Track, per pick and per day it appears: symbol,
scan date, support set, the score and rank, the chosen strike/expiry/premium. Grade at the
sold put's expiry and at 5/10/20 sessions: did price hold above the strike, max adverse
excursion in ATR, which supports broke first."*

The D1 scan has printed `master_avwap_theta_puts.txt` since Phase 0.11 and **nothing has
ever read a theta pick back**. There is no theta row in the setup tracker, no theta cohort
in the outcome store and no surface that answers "does a three-SMA stack actually hold a
sold strike". This packet adds the record and the grade. It adds no term to the theta score,
no support to a stack, no line to the report and no gate anywhere:
`scripts/master_avwap_lib/legacy.py` is READ by it and was not edited.

### Four things the real scan rows settled

A hand-written fixture would have got every one of these wrong, which is why the tests build
their rows through `evaluate_theta_put_candidate` / `evaluate_theta_pcs_candidate` and
`_apply_best_option_to_theta_row` rather than by hand.

- **`SMA_20` is BUILT as a theta support and then DROPPED.** `evaluate_theta_put_candidate`
  asks `_theta_support_entry` for SMA_20/50/100/200 and `_is_valid_theta_support_entry`
  (`legacy.py:21082`) then refuses SMA_20. So a recorded support set names SMA_50/100/200,
  and a "three-support" pick built from SMA_20/50/100 is not a pick at all - it fails the
  `THETA_MIN_SUPPORT_LEVELS` floor. **The store records what the ROW carries, never what the
  builder attempted.**
- **`held` is `level <= close` and is NEVER derived from `distance_atr`.**
  `_theta_support_entry` keeps a level up to `THETA_SUPPORT_ABOVE_TOL_ATR` (0.05 ATR) ABOVE
  the close and CLAMPS that negative distance to `distance_atr: 0.0`. A level 0.05 ATR
  overhead and a level sitting exactly on price therefore record the identical distance, and
  only the level separates them. A recorder that read the distance would mark an overhead
  band as a support that was holding.
- **A sold put carries `strike`; a put credit spread carries `short_strike` and
  `long_strike`.** Both shapes are recorded in full and `strike` is the SOLD leg either way,
  so a credit spread is never a NULL strike.
- **`_apply_best_option_to_theta_row` REPLACES `score` with the option's `rank_score`** and
  keeps the support score as `base_score` (87 and 35 on the tester's AAA row). The report
  ranks on the former, so the store records both and the grade line grades the former.

### The rules this produced

- **One row per `(symbol, scan_date, play_type)` in `theta_picks.jsonl`** (shared home,
  `project_paths.THETA_PICKS_FILE`), written from the RUNNER right after
  `write_theta_put_report` - the scan's own output pass, never `legacy.py`'s tracker save.
  A key already present is not rewritten, so a rerun or a deferred option pass cannot double
  the n. **A failed append loses the row, never the scan**, and one malformed entry in the
  list does not cost the good ones.
- **Every appearance is an observation; the cohort grain is the FIRST appearance.** A repeat
  day is its own row and keeps `first_seen_scan_date`. In the readout, `n` counts first
  appearances and **`repeat_days` sits beside it and is never summed into it** - a name the
  scan finds eight days running is ONE observation of the outcome and eight days of interest.
- **`closes_for` hands back `{date: {"close","low","high"}}`, not a bare close.** It keeps the
  name `session_horizon_outcomes` established, and the value shape is wider because the MAE
  and `first_support_broken` are questions a close-only series CANNOT answer: a session that
  cut through a support and closed back above it is a break the closes never see. The
  docstring says so.
- **The marks land on exchange SESSIONS.** A 2026-06-01 scan's 20th session is 2026-06-30,
  because Juneteenth (2026-06-19) is not one; twenty business days lands on 06-29. The
  fixture puts a BROKEN close on 06-29 and a HELD close on 06-30, so a weekday walk fails on
  the number rather than passing quietly.
- **Completed bars only, and a session the calendar has not reached is `pending`**, with
  `target_session_not_complete`, never a break. The expiry grade waits for the expiry session
  to complete: before it there is no verdict, not a loss.
- **Only a support that was HOLDING on the scan date can break**, and `first_support_broken`
  is `none` when nothing did - a word, because a blank reads as "we did not look" and the
  nearest support would read as a break that never happened. A tie inside one session goes
  to the highest level, the one price crossed first on the way down.
- **A pick with no option quote is `unmeasured`, never broken.** `option_status:
  no_weekly_options` leaves `best_option` empty, so there is no strike and "did price hold
  above the strike" has no answer; grading it False would score a play the trader was never
  offered.
- **RS is `not_measured` and never invented.** Recon found NO relative-strength term in
  today's theta scoring - `_theta_support_quality` takes source and `distance_atr` and
  nothing else - so the RS cut is a column that says so with a note naming the function.
- **Hold rate leads, the sort is the BOUND, and the grade line is the point system's.**
  The cells carry the ONE Wilson bound (`swing_headline`) and the floor is
  `evidence_stats.MIN_REPORTABLE_N`; the tercile line reuses `setup_points_evidence.Cell`, so
  the desk's two grade lines never say the same thing two ways, and it refuses under 30 per
  third with that module's own `not enough per third yet`.
- **The slot is APPENDED at the END of the deterministic stage.** `theta_pick_grading` sits
  directly after `daily_digest` and ahead of `ai_summary`; decision 0018 holds - a later
  phase appends inside its stage and never reorders across stages - so `EXPECTED_SLOT_ORDER`
  gained one name in that position and nothing above it moved. It is deterministic, calls no
  model, is idempotent, and it deliberately runs after the digest so a theta grade can never
  delay the fact pack.
- **The readout is built ONCE, on the Setup Tracker's read worker.** Its cells and both its
  sentences come off the same `theta_readout` call; the Qt thread renders them and computes
  neither. Building it twice would be two answers to one question.

### What is NOT here

No promotion, no score, no gate. The Theta tab is shadow evidence with a population sentence
above it, and 20 sessions after the first scan is the earliest any cell can carry an `n` and
a bound. The RS cut stays `not_measured` until someone adds an RS term to the theta score on
purpose; the readout will not invent one to fill a column.

## CH - the bars exist, the view is a window (2026-09-13, WISHLIST sweep)

### The complaint, and what it actually was

The trader's sentence is *"200 candles is not enough"* (WISHLIST 10H). The reflex reading is
"fetch more history", and it is wrong. `chart_snapshot.load_d1_bars` has always read the
**full** daily history out of the durable parquet store, and `build_d1_snapshot` has always
computed SMA50/100/200, EMA8/15/21 and the AVWAPE bands over that whole history **before**
slicing a display tail. The bars were already there. `D1_DEFAULT_SESSIONS = 90` was the
slice, and the slice was the complaint.

So this packet bought about four years of D1 history for one longer list slice of bars that
are already in memory, and **not one additional provider request**. The warm-up the trader
worried about ("enough warm-up before the visible span for EMA/LRSI/AVWAP") was never at
risk - it is what the module already did - and the golden fixture proves it: every overlay's
last 90 values are byte-identical to what the 90-session chart produced before the change.

### The two numbers, and why they are two

- `D1_HISTORY_SESSIONS = 1000` - how far back the payload REACHES.
- `D1_DEFAULT_SESSIONS = 90` - how many bars the chart OPENS on.

`CandleChart.set_data(..., initial_view_sessions=N)` holds every bar and frames the tail, so
**panning left reveals the older bars with no request of any kind**. `setClipToView(True)` +
`setDownsampling(auto=True, mode="peak")` were already set on every chart, with a comment
saying they resolve to a no-op at 90-500 bars and "earn their keep when a longer history is
zoomed into"; this is that. Measured offscreen at 1,000 candles with 14 overlays: `set_data`
24-31 ms, `grab()` 15-22 ms - a frame, not a freeze.

The history target is a TARGET, capped by what the store holds. The payload carries
`oldest_available` (the oldest bar DRAWN) and `history_truncated` (the store has more behind
it). 300 stored sessions report 300 and `False`: there is nothing further left to pan to, and
a strip that claimed otherwise would invite the trader to drag at a wall.

### Three things that would have been wrong

**A y-range from the payload.** The fixture walks from about 20 to about 197 on purpose. The
last 90 sessions live between 163.89 and 196.74; the 1,000-bar payload starts at 39.30. A
scale taken from the payload flattens today's candles into a line - the same chart the
trader already had, only wider and now useless. The y-range comes from the VISIBLE window.
The log/linear decision still asks EVERY bar, though: a non-positive bar off the left edge is
one pan away, and flipping the scale under the trader mid-drag is worse than opening linear.

**Levels from the payload.** `chart_levels.horizontal_levels` filters store levels to the
chart's price range and then applies a clutter budget per bucket. Handed a four-year range it
admits four-year-old levels, and they compete for that budget with the lines the trader can
see. So `build_d1_levels` gained `price_range_bars` and `ChartDataService` passes the INITIAL
VISIBLE window: today's behaviour, exactly (lead ruling, 2026-09-13). **Panning left does not
recompute levels** - the payload is fixed at build time and the paint path reads no caches.

**A provider request sized off the history target.**
`SymbolSnapshotWidget._start_d1_backfill` asks for `max(260, ceil(sessions * 365 / 252))`
calendar days when the store looks stale. Wiring 1,000 sessions into that field would make
every click on a stale symbol a 1,449-day Yahoo request. It is a repair for one symbol, not a
history import - the store is filled by the scan pipeline - so it still sizes off the host's
`d1_sessions` (260 compact, 754 for Chart Review) and a test caps it at 800.

### M5: the button, the merge, and the view

Two sessions on open, as always; **Load older** adds two, up to ten per symbol per desk
session. It reads the same in-memory `bot.m5_chart_bars(max_sessions=n)` the chart already
used - documented as a cache read that never fetches - so there is no new provider door. The
pan-left trigger the packet allowed was deliberately NOT wired: a pan that fetches is a fetch
on the paint path, and a test arms the fake bot to raise on any call during a `grab()`.

The chunks overlap by construction (a 4-session read contains the 2-session one), so the
merge is a CUT at the fresh chunk's first bar, not a set union: no bar can appear twice, the
order is the order it was already in, and a bot whose cache has since shrunk cannot take
history off a chart that has it.

**"Viewport preserved" means the same CANDLES, not the same index range.** Older bars arrive
on the LEFT, so restoring `(100, 140)` verbatim after 156 bars are prepended slides the
trader a session and a half back through their own chart. `CandleChart.visible_bar_span`
records the first and last candle's `dt` plus the y-range; `restore_bar_span` puts those
candles back and returns False rather than guess if either is gone.

A raising provider costs the older bars and never the chart: the extra sessions roll back to
what the chart actually reached, the drawn bars stay drawn, and the button reads `older bars
unavailable`. A result for a symbol the trader has left is dropped twice over - the service
already keeps only the newest request per symbol, and the render path checks the symbol - and
a symbol switch resets the count to two.

### What was not built, and why

**H1/H4 (packet item 3).** The desk draws neither. `bounce_bot_lib/legacy.py._closed_h1_bars`
aggregates completed H1 bars and `master_avwap_lib/legacy.py:28235
resample_intraday_bars_to_4h` resamples H4 from that H1 history; both feed the HTF trend/
retest/EMA15-rejection study, and neither reaches a chart, a widget or a payload. Every
`CandleChart.set_data` call in `scripts/ui/` passes `timeframe="d1"` or `"m5"` and nothing
else. The packet's own instruction was "if the desk does not draw H1/H4 today, say so and
stop at D1/M5". A 500-bar target is only meaningful once such a chart exists.

**The centre pane's own provenance line.** Item 4 asks for the oldest date in "the chart's
existing provenance strip". Exactly one exists - Chart Review's `provenance_state`, which now
appends `D1 back to <date>` and `(more behind)` when truncated. The centre Visual Alert
Review pane has no strip to add to; giving it one is a layout decision for the trader, not a
wiring one, and the arm bar's position rule says not to move that furniture uninvited.

### Reopen trigger

An H1 or H4 CHART appears on the desk (then item 3's 500-bar target becomes real work); the
daily store stops being the source of D1 bars; or the trader asks for the oldest-date readout
on the centre pane as well.


## WS - wrong side is shown, never hidden (2026-09-12, WISHLIST item 9)

**The trader, verbatim (WISHLIST item 9):** *"stop putting up longs below avwape and
shorts above it."* The lead's ruling for the sweep, which the trader may overrule
(`plan.md` Phase 0.26): **display only**. A `wrong side` chip on the Desk's setups row
and a ` [wrong side]` tag on the AWAY digest's swing line, on the CURRENT anchor only.
Hiding such a row is a DETECTOR decision - it changes which setups exist - and it is not
built. Nothing here reaches a score, a gate, an alert, a watchlist, Focus, the review
queue or `review_policy.json`, and nothing is re-ordered or filtered.

**What the row actually carries, measured.** The packet's premise was that the desk row
carries `current_close` and `current_avwape`. It does not. Those two are
`feature_snapshot` fields on the TRACKER's daily marks (`legacy.py:5132-5133`); the rows
the setups table and the AWAY digest read are SCAN rows, and on the live focus feed of
2026-09-12 **zero of 435** carried either field. Every one of the 435 carried
`current_band_zone` - top level or under `setup_candidate.trigger` - written by
`runner.py` from `legacy.get_band_context`, whose vocabulary is the ordered band levels
(`LOWER_3 .. LOWER_1`, `VWAP`, `UPPER_1 .. UPPER_3`). A zone names the two levels the
close sits between, so it answers the same question the two prices would, and reading it
needs no `legacy.py` edit (the packet forbade one).

**So the rule has two bases and says which it used** (`scripts/avwape_side.py`, pure - no
I/O, no clock, no Qt, because `paint` calls it). `wrong_side(side, close, avwape)` is the
packet's function: a LONG under the anchor or a SHORT over it, **tolerance 0** (a close
exactly on the line is the RIGHT side), `None` whenever the side or a number is missing.
`wrong_side_from_zone` is the same verdict from the zone. `read_row` prefers the numbers
and falls back to the zone, and `tooltip_text` **never prints a price it did not read** -
priced rows get `LONG below AVWAPE 412.50 (close 409.10)`, zone rows get `LONG below
AVWAPE (band zone LOWER_1 to VWAP)`.

**`favorite_zone` is deliberately not a fallback.** It names the zone the SETUP wants
(`"LOWER_1 to AVWAPE"`), not where price is; reading it as the close's position would
badge rows by their setup shape. `legacy._priority_current_band_zone` does fall back to
it - for a different question.

**The chip is a second pill, never a repaint of the first.** `SetupTableDelegate._chip`
now returns its rect and takes `after=`, so the `wrong side` pill starts a gap past the
bucket chip; `sizeHint` asks for that extra width, because `fit_columns` sizes a column by
measuring the delegate. Below `_MIN_CHIP_WIDTH` the second pill is **not drawn at all** -
the compact profile pins `bucket` at 96 px and a 12 px sliver of colour is not a badge -
and the tooltip still carries the whole sentence. The tooltip ADDS a line to the bucket
label the cell already showed; it never replaces it.

**On the digest the tag costs nothing.** It is computed after the ranking and after the
near cap, from `pick["raw"]` - the row WS-PT4 already put on the projection, so the digest
and the table are one reading of one row - and a reader that raises costs the tag and
never the digest (`is_wrong_side_row` swallows). On the 2026-09-12 feed the badge would
have flagged **87 LONGs under the anchor and 6 SHORTs over it, of 435**.

**Still the trader's to decide** (open, recorded here rather than guessed): whether the
wrong side should eventually be HIDDEN or demoted rather than badged; whether the PREVIOUS
anchor counts as well as the current one; whether the M5 alert list, the chart review pane
and the Focus surfaces should carry the same badge; and whether a wrong-side pick should
be excluded from the AWAY push.

**Tests:** `tests/test_ws_ws_wrong_side.py` - 53 of 57 red on the pre-build tip (the four
that passed are the "no chip" / "no tag" negatives, vacuously true before the feature),
58 green after. Two goldens were taken from the pre-change code and re-taken after: the
right-side bucket cell renders to the same image (sha1
`c6cecd5e2481dd4aa0589fce62466045a549095b`) and a digest with no wrong-side picks renders
byte-identical (sha1 `d6b287457a939368d311af6f7f74380ace523ab0`).

**Reopen trigger.** The trader asks for hiding or demotion; a scan row starts carrying
`current_close`/`current_avwape` (then the numbers basis becomes the live one and the
tooltip changes shape); a new band level joins the zone vocabulary.

## 10A - last scan, latest bar and shown report are three clocks (2026-09-12, WISHLIST sweep)

**The rule in CLAUDE.md.** Every Master AVWAP scan writes
`master_avwap_scan_manifest.json` and one append-only line to
`master_avwap_scan_manifest_history.jsonl` (`scripts/master_avwap_lib/scan_manifest.py`,
`project_paths` constants, temp + rename) recording three separate clocks: **when the scan
ran** (`started_at` / `finished_at`, aware market-local), **how fresh its INPUT bars were**
(`latest_input_bar_session` + `preview_bar_used`), and **what it published** (`outputs`, one
entry per file with its row count). `status` is `ok` only when the scan reached its whole
universe, `partial` when it RETURNED having fetched fewer symbols than `universe_size`, and
`failed` when it raised - and a failed scan **touches no output file**, so the last good
report keeps its bytes AND its mtime. `freshness_line` builds one sentence from that
manifest and the report's mtime; the Setups status row and the System Health check
`master_scan_freshness` print the SAME sentence. `scripts/master_avwap_lib/scan_replay.py`
replays one session for one symbol from the dated copies and reconstructs nothing.

**Why.** The trader's report (WISHLIST 10A): *"the strongest Master AVWAP updates seem to
arrive in the final hour or at EOD, even when the app says it updated earlier"*, and *"a
recent file timestamp proves neither fresh input bars nor good discovery"*. That is two
complaints wearing one coat, and the desk could answer neither. The Setups panel showed
`Last run: <stamp>`, written only by `_on_scan_finished` - so after a failed scan, or after
any restart, it said nothing at all about what was on screen, and the old report sat there
looking current. `Setups as of <date>` came from the report's own contents, not from the
scan. Nothing anywhere said whether the daily bars behind a 12:31 report were Thursday's
completed ones or today's forming preview. `run_result` was an in-memory dict (`runner.py`)
that died with the process.

**The three clocks, and why they are three.** A 12:31 report built from Thursday's bars is
correct and current; a 12:31 report built from a forming Friday bar is a preview; a 12:31
report that is really Tuesday's file, kept because Wednesday and Thursday both failed, is
a trap. One timestamp cannot tell those apart, and every single one of them is a different
trade. So the strip says all three:

```
Scan ok 12:31 · inputs through Thu 09-10 (D1 complete) · shown: 12:31 report
Scan ok 11:05 · inputs through Wed 11-25 · inputs: today preview · shown: 11:05 report
Scan partial 12:31 (940 of 1097 symbols) · inputs through Thu 09-10 (D1 complete) · shown: 12:31 report
Scan FAILED 13:02 · showing 12:31 report (stale)
Scan: not recorded yet · no report
```

**Which session counts as complete is the calendar's answer, not a date comparison.**
`input_bar_freshness` asks WS-FC1's `daily_bar_cache.last_completed_session` **once per
scan** (the 2026-09-03 freeze was an uncached calendar at 84% of the GIL samples; a
per-symbol question over 1,097 frames would put it back) and then classifies every frame
against that one answer. A frame whose newest row is dated an incomplete session sets
`preview_bar_used` and contributes its newest COMPLETED row instead - so a forming bar is a
labelled preview and never moves `latest_input_bar_session`, which is plan.md sec 5's
completed-bars rule spelled for a manifest. `market_calendar` models no early closes, so the
half day after Thanksgiving is a preview until 16:00 ET: conservative in the only direction
that matters.

**Two clocks, deliberately spelled apart.** `freshness_line` renders every stamp **in the
offset the stamp itself carries** and never re-converts it, so a manifest written at 12:31
New York prints 12:31 on a Pacific desk - "when did the scan run" is a question about the
desk. The replay's four checkpoint windows (`open < 11:00`, `midday [11:00, 15:00)`,
`final hour [15:00, 16:00)`, `close >= 16:00`) are the opposite question - where in the
MARKET's day a scan landed - and convert to **exchange** time for that reason alone.
Without the conversion a Pacific 09:58 stamp (12:58 in New York, squarely midday) would be
filed under the open, and the trader's own "it arrives in the final hour" claim would be
answered with the wrong checkpoint every day. Pinned by
`tests/test_ws_10a_scan_freshness_builder.py`.

**The dated copies exist because the desk kept none.** The priority report is overwritten by
every scan, so there was no historic snapshot for the replay to read and the discovery half
of the brief was unanswerable. The smallest useful copy per PUBLISHING scan now lands in
`%LOCALAPPDATA%\TradingBotV3\diagnostics\scan_reports\<YYYY-MM-DD>_<HHMM>.json` - run id,
status, `finished_at`, `latest_input_bar_session`, `preview_bar_used` and one
`{symbol, side, bucket}` row per published row. A FAILED scan gets no copy: a snapshot
listing zero rows would read to the replay as "the name was not in the report", which is a
reconstruction. The cap is `REPORT_COPY_RETENTION_SESSIONS` (30) counted as the 30 most
recent **distinct dates**, not the newest 30 files - counting files would let one busy
Friday evict the week before it, shortening the window exactly when there is most to look
at.

**The replay refuses rather than reconstructs.** A checkpoint with no recorded copy prints
`open · no recorded snapshot` and nothing else, and a day that recorded nothing prints
`delay unmeasured` instead of a number. Re-running today's scanner over an old session would
answer with today's data and today's code; the brief asks for the missing proof to be
labelled, not filled in. When both are recorded the summary names first eligibility (the
first snapshot carrying the name in ANY bucket), first publication (the first carrying it in
`favorite_setup`) and the delay in minutes - the number the trader is actually asking for.

**Never on paint.** The manifest is read by `_ScanFreshnessWorker` off the Qt thread, started
from the EXISTING `refresh_from_reports` path. The Qt thread does two `stat` calls first
(`_scan_freshness_signature`) and starts no worker when neither file moved, because the
report watcher fires that refresh several times around a scan and a second worker racing the
first is how "never on paint" quietly becomes "usually not on paint".

**`legacy.py` diff is ZERO.** This packet carried no trader yes for the ask-first file, so
the manifest is a new module called from `runner.run_master` - the wrapper that already owns
both the success and the failure branch. `record_scan` never raises: an evidence store may
not cost the thing it records (plan.md sec 5), and a scan that produced a good report must
not be turned into a failure by a full disk.

**Reopen trigger.** The trader asks for a fourth checkpoint or different window boundaries;
a second process starts publishing the priority report (the manifest assumes one writer per
report); the dated copies are asked to carry more than symbol/side/bucket, which is the
point at which their size stops being free.

## TM - a prompt is a slot, an answer is a dated row (2026-09-12, WISHLIST 10J)

**2026-09-14 follow-up, trader-authorized:** "the trade mentor stuff should be a pop
up box" and "I dont need to see these values I jsut want the AI to have access to
it", with lightweight context for VXX, RSP, USO, TLT, IWM, QQQ, SPY, XLB, XLC, XLE,
XLF, XLI, XLK, XLP, XLU, XLV and XLY. This supersedes the under-chart placement below.
The hosting seam in `alert_chart_review.py` is authorized by this request; alert
decisions and the arm bar are outside the change.

The new surface is one reusable modeless window per host. Scheduled prompts do not
take keyboard focus. Closing or pressing Escape skips and preserves a draft; expiry
and pause hide the window. Hidden market context is measurement, never trader text
or an AI opinion. Each snapshot states its own capture time, rules, source and
coverage; a response can be later and never backdates that snapshot. The two trend
horizons stay distinct. Missing, stale or invalid inputs produce unknown values,
never zero or an invented trend.

The first context version uses completed M5 bars for the last 30-minute percentage
change and direction, plus above/below session VWAP only when regular-session
volume coverage is complete. Daily context uses the five-session percentage change
and above/below SMA20, through the last completed session (including early closes).
This is a small trend summary, not a reconstruction of every chart pattern. The
raw journal retains the named scalar fields; the AI copy uses shared defaults and
columns/rows. A varied 17-symbol snapshot plus a short real journal note was
independently retained in 2,201 characters under the existing 3,000-character source
budget. Exceptionally long notes or distinct missing-data reasons still obey the existing source limits: an oversized row is excluded with an explicit banner.

Collection is bounded to those 17 symbols, off the Qt thread, with at most one
worker and no new recurring timer. Opening a prompt requests context; repeated opens
reuse the hourly M5 result and D1 results through the same completed session.
Existing usable cache data is preferred; missing coverage may use bounded Yahoo
batches, never a new per-symbol request loop or IB. Failures respect the same throttle. Submit
does not wait, fetch or call a model; a pending/failed context is explicit and does
not cost the journal write. A late result cannot change a saved note. The existing
nightly Market Journal source carries the small snapshot beside the raw words;
full bar arrays and images never enter this payload. Dedicated Mentor feedback
remains deferred.

The two data horizons fail independently: an unavailable M5 provider must not erase
good daily context, nor vice versa. A completed-session D1 cache outranks an empty or
stale local-cache response; otherwise the four missing/stale daily symbols would be
downloaded again at every hourly prompt. The AI projection preserves schema, rules,
sources, capture time and availability/reason, including a stale snapshot marker.
Shared column defaults may compress repeated status/date/reason fields, but decoding
must recover every original scalar. Acceptance tests use a real built snapshot and
a persisted note inside the 3,000-character evidence budget, not a smaller fabricated
snapshot.

The Trade Mentor is the first thing on this desk that INTERRUPTS. Everything else waits
to be looked at. That single fact decides almost every rule below, because an
interruption that is wrong is worse than no interruption at all, and because the thing it
collects — what the trader thought at 09:00, in their own words — cannot be reconstructed
afterwards from anything.

**A prompt is a SLOT, and a slot is a record.** `trade_mentor_schedule.slots_for_session`
is pure: no Qt, no I/O, no clock read. A session is a tuple of `MentorSlot(slot_id,
session, scheduled_at, kind, expires_at, post_close)` and `slot_id` is
`<session>-<HHMM>-<kind>`. `trade_mentor_slots.json` keys one record per `slot_id` with
`delivered_at`, `answered_at` and `skipped_reason`. Everything the trader asked for falls
out of that pair: a desk restart mid-hour re-shows the card (the hour is open and nobody
answered) without moving `delivered_at` or writing a second record; a timer that fired
late or a clock the OS corrected cannot duplicate an hour, because the identity is the
wall-clock slot and not the tick; a missed hour is recorded with its reason and never
asked again.

**The close that ends the hourly window is the REAL close.** `market_calendar.
session_close` models every session as 16:00 ET by design and every existing caller is
right to use it. Asking it here puts an 11:00 M5 read on the day after Thanksgiving, an
hour after the tape stopped. `market_early_close.session_close` is the one this reads, and
2026-11-27 therefore carries five slots (07, 08, 09, 10, 12) and no 11:00. The two D1
hours and the 10:00 trade check SURVIVE the close and are LABELLED `post_close`: the
trader asked for the noon D1 read on a short day in as many words, and the 10:00 check
asks about yesterday's trades, not today's tape.

**Pacific is a wall clock, not an offset.** `ZoneInfo("America/Los_Angeles")`. The 09:00
read is -08:00 on 6 March 2026 and -07:00 on 9 March; a `timezone(timedelta(hours=-8))`
implementation passes every test written in winter and files half the year's reads an
hour away from the tape they describe.

**An unanswered prompt is no observation, and a draft is not an answer.** Expiry is
`scheduled_at + 1 hour`, which on a normal session IS the next slot, so a backlog can
never form. Half-typed text goes to `trade_mentor_drafts.json` exactly as typed and is
never a journal row, never shown as a read, never counted. The alternative — treating
silence as "no view" — would put a hole in the record indistinguishable from a trader who
looked and had nothing to say.

**Four absences, four reasons.** `away` (Auto mode AWAY), `paused` (the trader's "Pause
today", a DAY and not a switch), `locked`, `idle` (more than `IDLE_GRACE_MINUTES` = 20
since the last input). One boolean "present" would make the coverage gap unreadable
months later, and these are the only four states anyone will have to explain. A trader
quietly watching charts is PRESENT: the comparison is strictly greater than the grace, and
an off-by-one there turns a chart-watching hour into a skipped one. Presence is measured
by `user_presence.idle_seconds` — `GetLastInputInfo` behind a function that returns `None`
off Windows or on failure — and `None` is treated as PRESENT, because missing data is
uncertainty and reading it as absence would silently switch the feature off. A locked
workstation is carried by the same number (input stops at the lock); no session-lock hook
was built, and `session_locked` is an injected callable so one can be added later without
touching a caller.

**"Read unchanged" is a new row, never a correction.** It restates the previous read at
the current time with `reaffirms` naming it, and `supersedes` asserted EMPTY. Superseding
would HIDE the 09:00 read behind the 11:00 one, and "my view has not changed for two
hours" would become indistinguishable from "I only ever said it once". The two fields
`mentor` and `reaffirms` are new on `market_journal.build_entry`, present and empty on
every other entry rather than absent; the packet expected an existing extra/metadata
field and there is none.

**A read carries THREE times and never blends them.** `mentor.scheduled_at` is the hour
that asked, `mentor.responded_at` is when the trader actually replied, and `created_at` is
the ledger's UTC stamp of the same instant. A reply typed at 09:12 cannot claim to
describe the market at 09:00.

**A defect this found:** `market_journal_service.write_entry` built the entry from the
caller's `now` and then appended without it, so `EvidenceLedger.append` stamped `event_at`
AND its own `session_date` from `datetime.now()`. An entry written with an explicit `now`
was filed under one date and stamped with another, and a reader narrowing the ledger by
session missed it entirely. No production caller passed `now` before this packet, which is
why it never showed; the Mentor's injected clock is what found it.

**The 10:00 check asks only what is missing, and "no stop" is never a zero.**
`trade_mentor_trade_check` walks the calendar to the PREVIOUS EXCHANGE SESSION (Monday
asks about Friday), reads `journal_store.JournalStore` and `journal_coverage` — not
`shared_journal_service()`, which is the market journal and was the packet's one wrong
premise — and asks per trade only the material fields it lacks: thesis (`notes`), stop
(`planned_stop`), target (no column exists anywhere, which is WHY the four states exist)
and setup (`setup_tags`). The four answer states `not_supplied` / `no_fixed_target` /
`not_remembered` / `not_applicable` stay distinct because "I had no plan" and "I had a
plan I cannot recall" are different facts about a trader. Answers are ANNOTATION rows
(`opportunity_events` under a new `RECALLED` type — a kind, not a schema migration) and
`planned_stop` is NEVER written from here: `0.0` reads downstream as a stop at zero and an
infinite R, and a number remembered the next morning is not the documented pre-entry plan
that column means. Every row carries `recalled_after_session = True` and the actual write
time. The task is capped at three trades and the remainder is a COUNT the Journal's
completeness view shows, never a fourth question and never silence. No broker coverage for
the session answers `journal not ready` and asks nothing — an empty questionnaire drawn
from an incomplete import is a lie about the session.

**The pop-up never takes focus and the arm bar never moves.** `WA_ShowWithoutActivating`,
no `setFocus`, no `raise_`, no `activateWindow`, not modal; `Ctrl+Enter` is scoped to
the text boxes, because a hidden widget competing with a live window shortcut makes
both bindings fail. The original WS-TM card occupied `AlertChartReview`'s layout
after the arm bar. The 2026-09-14 trader correction removes it from that layout:
it is a separate window and consumes no chart height, even while open.

**Phase 0.31 finishes the form help.** Each incomplete trade now has one raw answer box.
Pressing its local-AI button writes `RECALLED_RAW` first, then a `QRunnable` asks the existing
medium local model for a strict draft. Only missing fields are allowed; each answer needs an
exact source span, numbers need one named unit, and a control the trader already touched is
never overwritten. Save answers remains the only writer of `RECALLED`; planned-risk columns
remain unreachable. The last verified market-story narration supplies one bounded coaching
question. The popup opens at 900x820, has a 760x720 floor, remembers resize on every hide/close,
and no show path calls `adjustSize` over that choice.

**Ownership.** `MainWindow` owns the service (one timer, one state file, and the card's
host is built more than once), started in `showEvent` and stopped in `closeEvent` beside
`WorkingLatelyService`; `AlertChartReview` owns exactly one card and the always-available
"Give a read" button, in the EXISTING verb row. Settings owns the checkbox (default OFF,
persisted, independent of Auto in both directions), the DST-aware sentence and a "Pause
today" button that emits a REQUEST rather than writing the service's state.

**Tests:** `tests/test_ws_tm_trade_mentor.py` — original 36 were written red before WS-TM;
Phase 0.31 adds raw-first, validation and persistent-size coverage.

**Reopen trigger.** The trader asks for a different hour, a different grace, or for a
prompt to survive being away.

**TJ-9 (2026-09-19): the 09:00 card insists** - and it fired two of the three reopen
triggers above at once, a different hour and a prompt that survives being away. Branch
`claude/tj9-forced-trade-labels`, tip `ee35ae54`, merged `8077a758`. The live journal on
2026-09-18 held 215 trades, ONE confirmed setup tag, 151 `needs_review`, 33 `provisional`,
zero notes, zero planned stops and ONE `RECALLED` answer ever - every "which of my setups
pays" question was waiting on data nobody was being asked for. The trader: *"I want to be
forced to label my trades around 0900 as per trade mentor."* `TRADES_HOUR` moved 10 -> 9;
`_kind_for` follows the constant, so an early close (2026-11-27, 10:00 PT) now carries four
slots rather than five - the check sits inside the hourly window instead of being forced
into existence after the close. The 10:00 read went back to being an ordinary `m5` card.

Six rules bite, and five of them were written by reviews that reproduced the defect first.

**(a) A machine guess is not an answer.** `_FIELD_SOURCES["setup"]` read `setup_tags`
whatever its `tag_status`, so a bulk-tagger guess silently retired the one question the
task exists to ask; `_setup_is_answered` now reads the `tag_status` that `list_trades`
already joins.

**(b) A rejection is never a setup.** `journal_analytics` writes `vetoed:<code>` and
`passed:<codes>` *"prefixed so a rejection can never be mistaken for an endorsement in a
Tags column"*, and TJ-9 gave that sentence a second reader. On a copy of the live journal
the card offered `Setup: vetoed:too_extended_from_base (confirm)` on APTV, and one click
would have written it `confirmed`, where "My setups" counts it. The prefixes now have ONE
definition (`VETO_TAG_WORD`, `PASS_TAG_WORD`, `REJECTION_TAG_WORDS`, `is_rejection_tag`,
beside `LINK_TAG_PREFIX`) which both writers and the guess filter use, and `confirm_setup`
refuses a rejection at the WRITER as well, because that is what touches the trader-owned
table. `eligible_setup_names` splits the column on `;` and tests each top-level tag WHOLE
before splitting further - a pass writes all its reason codes inside one tag as
`passed:thin,extended` and `split_tags` splits on the comma, so filtering after that split
would have left `extended` standing alone as an eligible setup name. The auto-tagger's
output was proven byte-identical across the change: 1,198 rows, same sha256.

**The guess is filtered by SHAPE, not by a closed vocabulary.** Rejections, links and every
`<prefix>:<code>` shape are dropped, and if nothing survives there is no guess. A
closed-vocabulary filter was considered and refused: the provisional lane's names come from
the scanner's own `setup_family` values and the tester's fixtures pin `opening_drive` and
`earnings_gap`, none of which are in `valid_setup_claim_ids()` (28 ids) or
`ai_jobs.enrichment.setup_vocabulary()` (33 families), so enforcing one would have deleted
the lane and failed the tester's own assertions. What shipped instead: the card LISTS the
53-name vocabulary in a NON-EDITABLE combo with the guess preselected, so nothing outside
it can be written unless the trader leaves the guess as it is. **Showing the card writes
NOTHING** - the database's sha256 is identical before and after. Tighten to
closed-vocabulary only if the tester's fixtures are updated with it.

**(c) The session is the FIRST FILL's.** `rebuild_trades` writes `trade_date = closed_at or
opened_at`, so on a position held across sessions that column names the day it was CLOSED.
The live SMPL trade opened 2026-08-27 and closed 2026-09-18: reading `trade_date` called a
label written three weeks after the entry `same_session` - the one reading
`label_provenance` exists to make impossible - and a label written on the entry day
`recalled_after`. **116 of the journal's 216 trades** have an opened date that differs from
`trade_date`.

**(d) The ride is keyed on the SESSION, and it survives an absence.** Clearing on the hour
was what made an unanswered card expire into silence; clearing on the session boundary
keeps the old protection (a question about Friday can never be saved against Wednesday) and
drops the silence. The section was also built only for kind `m5_trades`, and only 09:00
carries that kind, so a 09:00 that was away, idle, locked or expired took the whole day's
questions with it and a trader who sat down at 11:00 was asked nothing. Any delivered slot
of the session now carries it while the check is owed - an unlabelled trade on the reviewed
session, or its statement not landed - and an answered check brings nothing back. AWAY needs
no test at that seam: the service records the absence and never emits `promptDue`.

What is protected from a rebuild is a section with ANSWER WIDGETS, so combos the trader had
already set survive; a one-line state is NOT, and that distinction is itself a defect the
second review found. Gating on label visibility meant that once the 09:00 card printed
`journal not ready`, every later slot returned early: the section never became the questions
that day - even after the morning retry had landed the fills - and the card went on printing
a freshness date that was no longer true. A not-ready line has nothing to lose by being
rebuilt, and its date is re-read every slot.

**(e) A label knows when it was made.** `claimed_before_entry` / `same_session` /
`recalled_after` come from one pure function over aware stamps; the cases that matter are
exactly the ones where the naive strings compare the wrong way round (a like written
`14:15+00:00` is sixteen minutes BEFORE a fill written `07:31-07:00`). The upsert rule, in
full: a caller with nothing to say and an UNCHANGED tag keeps the provenance a confirm
recorded; a caller that CHANGES the tag and says nothing gets a RECOMPUTED one through
`trade_origin.label_provenance` rather than a stale one; a save that CLEARS the tags clears
the provenance with them. **The known lossy case, to be named wherever the three
provenances are reported:** `accept_auto_tags` APPENDING a tag changes the tag, so it
downgrades a `claimed_before_entry` to `recalled_after`. `trade_origin._is_midnight` is
deliberately WIDER than `journal_trade_shape.is_date_only`, which asks the market-local
question only: the live DRAM assignment row is stored `2026-07-16T00:00:00-07:00`, which is
03:00 in New York. The wider test can only push a row toward `unmeasured`, and a plan
invented for a time that never happened is worse than `unmeasured`.

**(f) The morning retry CALLS the one owner.** `JournalImportService` - its own `QThread`,
Questrade only, already the desk's single refresh-chain owner - at most once a morning and
only when the night ended without an OK import. IBKR has no day leg and the card says so.
**The whole trade-check build sits inside ONE guard**, because the read above it is what
the trader is actually being interrupted for. Before the first review the kind test sat
OUTSIDE that guard and read `KIND_M5_TRADES` off the wrong module, so every Trade Mentor
prompt would have raised `AttributeError` in a Qt slot and no trade section had ever
appeared on a running desk - while 8,873 tests passed, because no test drove
`_show_trade_mentor_prompt`.

**Three advisories, batched and NOT repaired.** `_trade_check_is_owed` reads the 390 KB
annotation log plus two `JournalStore` opens on the Qt thread on every prompt (~50 ms
today); it wants a per-session cache before the log grows. `_journal_retry_date` is
in-memory only, so a restart before 10:00 allows a second morning pull. And one guard test
passes on un-fixed code.

**Item 7 left this packet.** The Questrade instrument work became packet TJ-9Q - see
"TJ-9Q - a sold put is recorded backwards" below.

### 2026-09-19, TJ-14A: the card splits in two, and the internals stop being invisible

Branch `claude/tj14a-mentor-card` (tip `6808abd9`), merged `e8c04f88` into
`lead/p033-integration2`; reaching `main` after the night's AI run. Two reviews driven
through the real Qt slot, NO-GO then GO. plan.md 12.4 TJ-14 items 1 and 6; items 2-5 are
TJ-14B. **The lead has confirmed this vocabulary as TJ-10's and TJ-16's contract**, so a
later packet reads these fields rather than inventing its own.

The trader's words: *"For trade mentor make sure we differentiate predictions from just
'describe the market and your thoughts'! The hope is an AI can pickup on my tendencies and
what leads to good predictions and what leads to wrong ones"* and *"trade mentor should
automatically be processing what's going on with the internals we watch. RSP VXX USO TLT
and the sector ETFs XLK XLE etc. so the AI already has that"*. The measurement behind the
first: `market_thesis.extract_thesis` over the trader's 42 real notes gave 21 `unstated`, 8
bullish, 7 neutral, 6 bearish, and read *"D1 SPY is still downtrending"* as `unstated` - the
trader describes the tape far more often than they predict it, so the graded thing becomes
a CLICK and the words become context.

**What I see / What I expect.** `mentor.observation` holds the words and the entry's `text`
stays those same words, so every existing reader keeps working. `mentor.prediction` is
`{direction, horizon, confidence, because, schema: "mentor_prediction_v1"}`. Horizons:
`rest_of_day` on every card, `next_5_sessions` on the 08:00 and 12:00 D1 cards only - a
swing call offered six times a day would be the same click about the same five sessions.
Directions are `up / down / chop / no_view` for the day and `up / down / range / no_view`
for the five sessions; `no_view` is a COMPLETE answer, is never graded, and takes `How
sure` off the card. `How sure` is FORCED on every other direction (decision 0021 answer 29:
a prediction IS direction, horizon and confidence - TJ-16 reads calibration BY confidence,
so a call filed without one could never join it). `Because…` is the only optional part.
`market_journal.prediction_of` is the ONE reader and answers `None` for all FOUR older row
vintages in the live ledger (69 rows: 28 with no `mentor` key, 13 with `mentor == {}`, 6
without context, 22 full v1), so an extracted stance is never pooled with a clicked one.

**A row's timeframe and its prediction's horizon always agree, and the WRITER enforces
it.** The defect that made it a rule (review round 1): `read_unchanged` on the 09:00 card,
handed the 08:00 card's D1 row as "your last read" because `_previous_mentor_read` answered
`rows[-1]` and an answered `m5_d1` card writes M5 then D1, kept `previous.timeframe` while
the horizon silently fell back to the only one the 09:00 card shows - filing a
**D1-timeframe entry carrying a `rest_of_day` call**, a row true of neither timeframe and
permanent in an append-only ledger. Now `market_journal._check_prediction_timeframe` is
asked at both gates every write passes (`build_entry` raises `PredictionTimeframeError`,
`is_publishable` refuses the same pair for a row assembled as a dict literal); the host
hands the card the latest read of EACH timeframe (`{"M5": row, "D1": row}`, today's session
only); `Read unchanged` files one row per timeframe the card shows, each reaffirming THAT
timeframe's words with THAT horizon's fresh click, all or nothing per card; and a timeframe
with no earlier read makes the verb unavailable with the tooltip saying which one. The
silent fallback is deleted, not narrowed. A clicks-only answer is never offered for
reaffirmation, because the verb restates WORDS.

**Forced means the verb refuses, not just the button.** The gate is inside `submit()` and
`read_unchanged()`, because `Ctrl+Enter` and any future host call reach the verb without
touching a button; the refusal names the row that is still open. Submit stays grey until
every direction row AND its `How sure` are clicked. `skip()` is untouched and TJ-9's
trade-check Save gate is entirely separate - a prediction never arms it and a missing
prediction never greys it. `Read unchanged` demands a FRESH click: copying the 09:00 call
onto the 11:00 row would manufacture a prediction the trader never made, and the two hours
would always agree.

**A clicks-only row is a complete answer and its `text` is empty on purpose.**
`is_publishable` is relaxed for exactly one case - an entry whose `mentor.prediction`
carries a clicked direction - and every other empty entry is refused as before. Nothing
writes a sentence nobody wrote: `market_journal.prediction_line` words the call for a
SCREEN only, `market_story._entry_row` carries the structured call so the day pack never
re-derives a stance from words that are not there, and `entries_about`, `extract_thesis`
(`unstated`, no crash) and the bounded AI package were checked against a real wordless row
and needed no change. An `m5_d1` card answered with two clicks and no words writes TWO rows
in the same second with the same empty text, so `entry_id` takes a SALT (`timeframe|
horizon`) for a WORDLESS row only - an entry with words keeps byte-identical ids. An
`m5_d1` card always writes BOTH rows, and drafts keep unsaved clicks.

**`trade_mentor_context_v2`.** `XLRE` joins `SYMBOLS` in alphabetical place (18 symbols;
every existing index, including SPY at 6, is unchanged) because the desk's own
`group_rrs.SECTOR_ETFS` always had eleven SPDRs while the Mentor context had ten. Each
symbol gains the day's change (against the PRIOR SESSION'S CLOSE, so an opening gap is part
of the move), its place in the day's range (0-1) and the side of the prior session's high
and low. The `derived` block is the read the trader used to type by hand: `breadth` (RSP
minus SPY on the day), `fear` (VXX against SPY with a `divergence` flag when both go the
same way), `rates` (TLT), `oil` (USO), `sector_leaders` / `sector_laggards` (top and bottom
three, on the day AND over 30 minutes - two rankings, never one list read twice, ties
broken by symbol), `offense_vs_defense` (XLK/XLY/XLC against XLP/XLU/XLV) and
`sectors_above_vwap` (a count WITH its denominator). Every line names the readings it rests
on and a missing input makes THAT line `unmeasured` naming the input that is actually
missing - never zero, and never a line poisoned by a reading it does not name. Completed
bars only: a forming bar moves nothing, proven by a fixture whose forming bar closes at
500.00 against a 100.00 tape. v1 rows stay readable, and `compact_for_ai` projects both
vintages while carrying ONE scalar `common.internals`, because `ai_summary._bounded` cuts
SIX levels down - exactly where a derived line's `inputs` and leader lists sit - and the
model would otherwise have been handed `"[nested content omitted]"` where the sector names
should be.

**One builder, and a thin loader beside it.** `build_context` serves the live card and the
pure `internals_at(session, stamp, bars)`, which takes its bars as an argument - two
implementations of the same reading drift on the first rounding decision. `internals_bars_at`
is the only part of the module that touches a store: M5 from the durable Day Review tape,
D1 from `d1_environment_store._cached_daily_bars` (the same daily cache the live context
service reads) and, for a symbol that cache has NEVER held, one daily bar per prior session
built from the tape itself (`_session_bar_from_tape`, up to `_TAPE_D1_LOOKBACK` = 3
sessions back, no network, point-in-time). That second source is not an optimisation:
**RSP, USO and TLT have no file in `%LOCALAPPDATA%\TradingBotV3\machine_cache\daily_bars`**
- the scan universe never fetches them and the live card only ever escaped through
`trade_mentor_context_service`'s Yahoo top-up - so without it a rebuild could measure
neither breadth nor rates nor oil, three of the eight derived lines, for every hour the
trader never answered. A symbol answered from the tape has its DAY facts measured and its
five-session / SMA20 facts honestly `unmeasured` with the reason; `sources.d1` says
`daily_cache+day_review_tape` when it happened. The point-in-time cut is the builder's own
(`_valid_m5` / `_valid_d1` at the stamp), so no caller can widen it; a symbol missing from
both stores stays `unmeasured`. `internals_bars_at` has NO production caller yet - it is a
capability until TJ-10 and TJ-16 call it.

`day_review_bars.decided_symbols` adds the internals to the one batched post-close download
- **the symbol list and nothing else, no D1 leg, zero IB**. That download is one batched
yfinance call per FIFTY symbols (`CHUNK_SIZE`), and its fixed base grew from 4 names to 18,
so a session with more than 32 OTHER decided names now needs a second call where it used to
need one. (An earlier note said ~36; the arithmetic on the de-duplicated union is 50 − 18 =
32, and the four old benchmarks are all inside the 18.)

**The strip.** `MentorInternalsStrip` sits above **What I see**, is built from the context
the service ALREADY delivered (no second `request_context`, no fetch, no loader), is worded
by the pure `trade_mentor_context.internals_lines`, and is styled from `theme.qss` by
object name - no widget-level stylesheet, because a card that comes up every hour must not
parse CSS on the Qt thread. An `unmeasured` line is PRINTED as unmeasured: a blank where a
reading should be reads as calm.

**Measured, 2026-09-19 (offscreen, staged scratch home).** Card construct p50 0.60 -> 1.12
ms; `show_slot` p50 0.06 -> 0.13 ms; `internals_lines` 0.008 ms and `_render_internals` p50
0.010 ms - the strip is one `setText`. The stored Mentor row grew about 53-57% (4,462 ->
~7-8 KB on the popup-context fixture; roughly 7 answers a day), reading a synthesised
21-day v2 month file takes ~9 ms, and `_previous_mentor_read` costs ~10.7 ms once per
prompt on the Qt thread. The 09:00 card - strip, What I see, the prediction row and TJ-9's
forced trade section - is 591 px tall at 520 px wide inside its scroll area, nothing
clipped. Two test thresholds were widened with the measured number beside each (popup
context cap 6 -> 10 KB, resilience budget 3,000 -> 6,000 chars) rather than the guards being
dropped, and 6 WS-TM test functions gained clicks across 9 call sites.

**Advisories, recorded and not repaired.** A rebuilt row's `m5_as_of` is Pacific-zoned
where the live card's is Eastern - the same moment, to be compared with `astimezone` and
never as a string. `read_unchanged` now returns `{"ok", "entries": [...]}` rather than one
entry, so a host reading `result["entry"]` reads nothing.

---

## TJ-14B - the card asks for what the desk is missing, and nothing else (2026-09-20, packet TJ-14B)

The trader's words (2026-09-19): *"We don't need to run every question every hour but if we
need more data make trade mentor ask me for it. I'm happy to click boxes or give my
responses but then I expect the AI to take it from there."* TJ-14 items 2-5. Branch
`claude/tj14b-mentor-questions`, tip `cfe137d1`, merged `161e905c` into
`lead/p033-integration2`; not on `main`.

**The registry.** `scripts/mentor_questions.py` is PURE - no store, no Qt, no clock; every
lane arrives in the `state` mapping the caller built, because a trigger that opened the
journal would be a second opinion about it and a read on whatever thread built the card.
Each `QuestionKind` declares its trigger (a measured gap, never a clock alone), its click
options, the store it `writes`, its `cadence` / `expiry`, its `priority` and - the point of
the whole thing - its `consumer` and the `answer_key` that consumer reads.
`consumer_report()` resolves the dotted consumer and statically asks whether that reader
touches the key. **The probe is an `ast` walk of the consumer's SOURCE and deliberately NOT
a call**: the key counts only as a string constant used in CODE (a subscript, a call
argument, a comparison, a keyword value), so a comment, a docstring and a bare string
statement are all refused - and a probe that CALLED the consumer and looked for the value in
its output would pass for `json.dumps`, which imports, is callable, and will never read a
Mentor answer. Both of the reviewer's foolers are now tests (`tests/test_tj14b_registry.py`).

**A question is ASKED only when its answer has a reader** (decision 0021 answer 28; lead
decision, 2026-09-19). FOUR kinds ship DORMANT with the packet that builds their reader:
`trade_origin` and `open_position_check` wait on **TJ-12** (the Process line and the
long-hold rows - `trade_origin.planned_state` has no reader outside its own module, and
`walkaway_day.LONG_HOLD_SESSIONS` only sets a `not_judged_reason` on a CLOSED trade),
`grader_gap` waits on **TJ-10** (nothing emitted `needs_trader_input` when the packet was
written; TJ-10 has since merged on this branch and the field exists, so waking it is a small
follow-up), and `quick_like_followup` waits on **TJ-14C** - the fourth, found in the review:
its answer is an `opportunity_events` row and its named consumer
`ui.annotations.like_cohort.like_pick_rows` reads `claimed_setup_id` only off
`trader_annotations.jsonl` rows, so the KEY is genuinely read, the STORE is not joined, and
the live log holds 46 quick likes - the trader really would have been asked. `pending()`
never returns a dormant kind on a live card; `consumer_report` reports it as `dormant` with
the packet named. **No shim reader was written**: a reader nobody calls is the same lie the
walk exists to catch. Waking one is a one-field edit (`dormant_until=""`) plus the lane in
`MainWindow._mentor_question_state`, which already names all five of them.

**The budget of three.** Beyond TJ-14A's forced prediction rows and TJ-9's forced
`trade_label` section - both registered, both `budgeted=False` - a card carries at most
three questions, ranked by `priority` then kind. The remainder is COUNTED on the card ("2
more waiting") and carried on the window to the next card: never dropped, never a fourth. A
carried subject whose lane arrives empty next hour is still owed - a question leaves only
when it is answered or retired. Cadence decides "answered": `once` forever, `weekly` per
exchange week, `daily` per session, `per_card` never. AWAY returns an empty card - forced
rows included, because a card nobody saw is not a question.

**`Stop asking this` retires ONE subject.** `TradeMentorService.stop_asking` is the single
writer, `retired_subjects()` the reader, both persisted in `trade_mentor_slots.json` beside
the slot records. That file lives under `PERSISTENT_DATA_DIR` (`C:\TradingBotData`), not
`%LOCALAPPDATA%`, so the retired subjects and the day's pull tally survive a machine-cache
wipe as well as a restart. It is never stored as an ANSWER: filing `stop_asking_this` where
a setup name belongs would count it forever.

**Fills by day, and the ONE owner.** `mentor_questions.pre_card_pull` holds the whole policy
- `PULLS_PER_DAY_CAP` = 3 (a normal session carries six cards, so a cap of six is no cap),
`PULL_FAILURES_PER_DAY_CAP` = 2, `PRE_CARD_PULL_DAYS` = 2 (today, for a fill made this
morning, and yesterday for a late post), a tally keyed on the DAY that resets on the next
one, a corrupt tally read as EMPTY and rewritten clean, AWAY refused, and a service that is
already running spending the attempt without counting a failure. It CALLS
`JournalImportService` (its own `QThread`, Questrade only, the desk's single caller of the
refresh chain) and **never refreshes a token**. It never raises: the card the trader is
being interrupted for is worth more than the fills it wanted, and a fill a late pull lands
is asked about on the NEXT card. TJ-9's `morning_import_retry` now goes THROUGH it and keeps
its own once-a-morning rule, whose date is parked in the same persisted tally - which closes
TJ-9's advisory that `_journal_retry_date` lived only in memory, where a restart before
10:00 allowed a second morning pull. **IBKR has no day leg at all** and the card still says
so. *Was an hourly Questrade pull safe?* The chain rotates on every refresh and two
consumers on one machine snapped it on 2026-08-25, but `_authorized_get` refreshes only when
the access token is missing or expired or a 401 arrives AND the stored token has not already
moved - so N pulls a day is not N rotations. A small capped number is safe; the cap is the
conservative shape either way, and the failure cap stops the day rather than trying once
more. **There was no per-day failure cap anywhere in the repository before this packet** -
`journal_importers.refresh_access_token` has the lock, the re-read inside it and the one
atomic save, and no counter of any kind; the only cap that existed was `ai_jobs/runner.py`'s
`max_attempts=3` on the nightly `journal_import` SLOT, which is one night's retry budget.

**ONE card starts AT MOST ONE import, and the three attempts are RESERVED and SPACED.** When
TJ-9's three-day morning catch-up is owed (the reviewed session's statement has not landed
and nothing was retried today) it goes FIRST and the pre-card pull is skipped on that card -
three days cover today too - and it does not spend the pre-card cap. `last_retry` is stamped
**only when an import actually started**: a refused or busy start leaves the morning owed.
Otherwise the pre-card pull runs, and only on one of the day's three reserved cards
(`pull_slot_ids`): the 09:00 card (the trade check's own hour), the middle card between it
and the close, and the LAST card - read off the session's REAL slot list, so an early close
has no middle card and simply forfeits that attempt rather than rolling it earlier.
Measured through the real `MainWindow._show_trade_mentor_prompt` with the reviewed session
uncovered: the six slots of 2026-09-14 (07 / 08 / 09 / 10 / 11 / 12 Pacific) call the import
service with days `[3, 2, 2, 2]` - one uncapped catch-up at 07:00, then the three pre-card
pulls at 09:00, 11:00 and 12:00; the four slots of the 2026-11-27 early close (07, 08, 09,
12) start two pulls, the middle attempt forfeited. The wording is **three pre-card plus one
catch-up**.

**Same-session fills, and the merge rule.** `build_task` lists today's unlabelled fills
beside the reviewed session's, deliberately NOT gated on import coverage - today's statement
does not exist yet, and waiting for it is exactly how every live label came to be
`recalled_after`. `save_answers` stamps `label_provenance` from
`trade_origin.label_provenance` (asked with no setup, so only `same_session` and
`recalled_after` are reachable; `claimed_before_entry` stays `confirm_setup`'s to decide) and
writes the BOOLEAN that matches instead of the hard-coded `recalled_after_session: True`
every row used to carry. **A DATE-ONLY first fill is never `same_session`**: it reads
`recalled_after` with `label_provenance_reason` saying why - a broker file is authoritative
for money and blind to time, so there is no moment for a label to have been made before.
Every delivered slot of the session hands the card a FRESH `TradeCheckTask` and the card
MERGES it: a trade row already on the card keeps its exact widget objects and their current
values (never rebuilt), a trade the fresh task names and the card does not is ADDED after
them, a row that is no longer owed is dropped, the heading and the freshness line are ALWAYS
rewritten, and the Save gate is recomputed over every row now on the card. Each trade's
widgets live in ONE container so a single trade can join or leave without touching another,
and every block NAMES its session (`AAA LONG - today (2026-09-14)`, `BBB SHORT -
2026-09-11`) - because `same_session` versus `recalled_after` is the whole point of asking on
the day, and a row that says only the symbol cannot show it.

**The AI question.** `NARRATION_JSON_SCHEMA` gains `mentor_question_options` (array,
`maxItems` 4, string items, `maxLength` 60 - nowhere near the 2,000 behind gate #144) and the
narration is VALIDATED against it: a five-option output is rejected WHOLE
(`degraded_no_narrative`) and the last verified file stays byte-identical. The field is
OPTIONAL, so a night that emits none is unaffected. On the card the overnight question stops
being the `One thing to test: ...` line printed on every card forever with no options and no
answer: it becomes ONE click a day whose answer is a dated Market Journal row, and the legacy
line is hidden whenever the click is offered so it is never asked twice.

**The three review rounds: NO-GO, NO-GO, GO.** All three were driven by reproduction through
the real Qt slot.

* **Round 1 (NO-GO, three blockers).** (1) The two-day pre-card pull started first, the
  import service's `running` guard refused the three-day morning catch-up, and
  `morning_import_retry` stamped `last_retry` anyway - so a Monday whose Friday-night import
  failed never reached back to Friday. (2) First-come spent the whole day's budget by the
  08:00 card, so no fill after 11:00 ET was ever imported and `same_session` - the label the
  packet exists to make reachable - was unreachable all afternoon while the card kept asking.
  (3) `quick_like_followup` named a consumer that reads its key off another store: the fourth
  dormant kind. The round also produced the AST probe (the reviewer's two foolers), the
  corrupt-tally rule, building the trade section BEFORE the pull path so nothing there can
  cost TJ-9's forced questions, `auto_mode` passed into `morning_import_retry` so AWAY
  refuses at both seams, and the like lane bounded to the sessions a question can be about
  instead of walking the whole append-only annotation log on the Qt thread every prompt.
* **Round 2 (NO-GO, one blocker).** `set_trade_check` returned early on a not-ready journal
  and drew NOTHING - including for the fills the desk had already SEEN today, which are
  exactly the mornings `same_session` matters most on. The question widgets became one shared
  builder used by both branches; a not-ready card with nothing seen today still says
  `journal not ready` and asks nothing, byte-identical to before.
* **Round 3 (NO-GO, one blocker, then GO at `cfe137d1`).** Once the 09:00 not-ready card drew
  answer widgets for today's own fills, `open_answers_session()` answered with the session
  and `MainWindow._show_trade_mentor_prompt` returned before `set_trade_check` on every later
  slot - only 09:00 carries kind `m5_trades` - so the REVIEWED session's trades were never
  asked about that day and the card kept printing `journal not ready` with a freshness date
  that had gone stale. The early return protected the trader's half-set widgets and nothing
  else was allowed to move because of it; the protection now lives in the MERGE, row by row,
  where it cannot also freeze the words above the rows.

**Item 5 - "the AI takes it from there": the manual-step audit.** *Automated or removed by
this packet:* answering the overnight coaching question (a sentence printed on every card
with no path to a stored row at all, now one click a day); labelling a fill on the day it
happened (`build_task` only ever reviewed `previous_exchange_session`, so the middle
provenance was unreachable and every live label was `recalled_after`); stopping a question
the trader does not want (there was no mechanism - the only way was to keep ignoring it);
and looking for the day's fills (the only day-time pull was TJ-9's morning retry, and only
after a failed import). *Kept, already automated:* TJ-9's morning retry, now routed through
the one owner rather than beside it. *Manual BY DESIGN and staying manual:* the
`review_policy` sign-off; `python -m ai_jobs.digest approve-audit` (gate #144 / Q4's second
half); the **Questrade token repair** - the trader pastes a token and `refresh_access_token`
never clears a rejected one, so TJ-14B deliberately stops pulling after two failures a day
rather than trying to repair it; and a CONFIRM of a setup tag, which is the trader's click
through the Journal's own writer (TJ-14B's quick-like follow-up writes no annotation row at
all). *Findings rather than defects:* the four dormant kinds - their answers are not asked
for, rather than asked for and dropped - and **`confirm_setup` can still stamp `same_session`
for a date-only fill**. TJ-14B fixed that in `save_answers` by refusing `same_session` when
`trade_origin.first_fill_at` is `None`; `confirm_setup` calls `trade_origin.label_provenance`
directly and was left alone, because `trade_origin.py` is TJ-9's pure rule and changing it
moves every caller at once. The rule belongs in `label_provenance` itself - one line for
TJ-12 or a follow-up.

**Lead decisions ratified 2026-09-20, each the trader's to overrule.** (a) The day's pre-card
import pulls are THREE, reserved for the 09:00, 11:00 and last cards, plus ONE uncapped
morning catch-up that goes first - the reviewer measured every pull spent by 08:00 otherwise.
(b) The card MERGES a fresh task, so a not-ready morning becomes yesterday's questions while
half-set widgets survive - freezing the section to protect the widgets also froze the words
above them. (c) A same-day answer is labelled `same_session` and a date-only fill never is -
the middle provenance was unreachable on every live row, and a broker file has no moment a
label could have been made before.

Live gate **#159**'s TJ-14B clauses and the new gate **#163** are owed, on real sessions.

---

## TJ-9Q - a sold put is recorded backwards, and the type is never read (2026-09-19, split out of TJ-9)

Measured by the TJ-9 tester on a READ-ONLY copy of `trade_journal.sqlite3`, which refuted
the packet's premise that item 7 was additive, so it was removed from TJ-9 and written as
its own packet (`.claude/packets/TJ-9Q.md`, branch `claude/tj9q-questrade-instrument`,
which holds the tester's parked tests). Nothing below is built.

**The type is never sent.** Questrade's executions endpoint carries no `securityType` or
`symbolType`, so **226 of 226 live Questrade fills are `security_type='UNKNOWN'`**. The
instrument IS readable from the payload's own symbol (`AAOI18Jun26P120.00`) and side
(`STO` / `BTO` / `BTC` / `STC`).

**Why it is not additive.** `journal_identity.group_key` and
`journal_store._contract_multiplier` both READ the type. A forward-only classifier would
split every open position from its closing fill, and a backfill alone would multiply option
P&L by 100. Classify, backfill and rebuild are therefore ONE tested CLI step, dry run by
default - **and the live `--apply` is the TRADER's act**, never an agent's.

**And the side map is wrong today.** `journal_importers.normalize_side` does not map
`STO` / `BTC` / `Cov`, so `_signed_quantity` reads a sold put as `+qty`: the trader's four
sold puts read `direction='LONG'` and three of them (AAOI, BE, QBTS) sit stuck `OPEN`. For a
put-seller whose whole strategy is selling premium, the journal currently records the
opposite of what he does.

---

## 10C - a watch is entry timing, never a claim (2026-09-13, WISHLIST sweep)

- **What the trader said (WISHLIST 10C):** on selected weekly-pattern names, *wait for a
  better entry* - a quick H1/H4 retester arm button below the chart, on the shared arm
  surface. And the fence around it, in the same breath: **a like or a tag alone never
  arms it, and the watch expresses entry timing, not a setup claim or an order.** Step 1
  built first as the H1 15-EMA bounce. Phase 0.31 adds step 3's completed-D1 frozen
  trendline break/retest/confirmation watch; step 2 (H4/LRSI) remains excluded.
- **Why that fence is load-bearing.** Everything else this desk records is a claim of
  some kind - a like is training data, a veto is a verdict, a Focus pick is a name worth
  watching. A retester watch is none of those: it says *"tell me when this shape prints,
  I will decide then."* So it grades nothing, joins no cohort, reaches no detector,
  score, tier, gate, watchlist, Focus list, review queue or `review_policy.json`, and
  writes no `pick_feedback` verdict. It fires once, it disarms, and the decision log
  gets `arm_watch` / `watch_fired` / `watch_invalidated` / `armed_alert_expired` rows
  that the review scoreboard already ignores by name (`review_learning`'s
  deliberately-not-added list). **A pattern firing is not proof of a profitable entry;
  usefulness is measured later.**
- **The rule sheet is frozen first, then wired.** `h1_ema_bounce_v1` lives in
  `scripts/indicators/h1_ema_bounce.py` and its full table is in
  `docs/M5_SIGNAL_ENGINES_PLAN.md` section 10: touch within 0.25 ATR of the 15-EMA,
  reclaim on the LAST completed bar by 0.10 ATR, inside 3 bars, EMA sloping the trade's
  way over 5, invalidated by a close 1 ATR through the line, `ambiguous` when a single
  candle does both, 45-bar warm-up, 24 h staleness, invalid candles skipped and counted.
  A change to any of those is a NEW version beside it, because a fired row carries the
  version it was measured under.
- **Why the H1 aggregation is a copy.** `bounce_bot_lib.legacy._closed_h1_bars` is the
  shipped rule and the packet asked for it to be reused. It could not be: importing that
  module drags `ibapi` and ~1,050 modules (2.55 s, measured 2026-09-13) into what is
  meant to be a pure indicator, and it takes `IbBar` objects rather than the dict bars
  `m5_chart_bars` returns. So `closed_h1_bars` is a faithful copy over dicts, the golden
  pins it **bar-for-bar** against the original, and a subprocess probe asserts neither
  `ibapi` nor the engine package loads. Nothing in `bounce_bot_lib` is edited and
  `H1_ALERTS_RETIRED` keeps exactly its four mentions: the retired H1 emitters stay
  retired, and this packet adds a watch, not an emitter.
- **One kind on this surface is not session-scoped, and it takes TWO readers to mean it.**
  `chart_watch.PERSISTENT_WATCH_KINDS` is the single name. `load_chart_watches` keeps
  those rows when the file's market date does not match (every other kind still dies at
  the roll, which is what `new_hod` beside it is for), and `watch_is_stale` returns False
  for them - without that second reader the watch survives the restart and the 30 s M5
  poll deletes it a minute later, which is the same bug with a longer fuse. Its life is
  10 TRADING days through `armed_alert_expiry`, counted on the exchange calendar, and
  uncertainty never deletes.
- **`armed_at` stays NAIVE.** It is the chart-watch store's own convention - IB serves
  this desk's bars on the local clock and arm times come from the same clock - and both
  `watch_is_stale` and `_evaluate_extreme` depend on it. The rows that LEAVE the store
  (`fired`, `invalidated`, the expiry row) carry the bar times the rule measured,
  attached to that same clock; nothing strips an offset off an aware stamp.
- **A new arm never fires on an old bounce** (repair 2026-09-13, review blocker B2).
  `h1_ema_bounce_v1` anchors its verdict at the LAST completed bar and knows nothing
  about arm times, so a series that already held a finished reclaim fired the instant the
  trader armed - on a move that was over before they pressed the button (the review's
  reproduction: golden confirm bar 11:30-12:30, `armed_at` 13:30, one poll -> one alert,
  watch disarmed). **Warm-up keeps every bar** - the EMA and the ATR are still computed
  over the whole series - and what is fenced is the EVENT: the rule's `confirm_bar_dt`
  (the reclaim bar for a confirmation, the closing-through bar for an invalidation) is
  eligible only when its END is strictly after `armed_at`, which is the existing
  armed-watch convention (`_evaluate_extreme`: `_bar_end(bar) <= armed_at` is pre-arm),
  inclusive on the pre-arm side; the bar end is `h1_history.h1_bucket_end`, so the short
  12:30 bucket ends at the bell. A candle that was FORMING when the button was pressed is
  post-arm once it completes, the same courtesy the M5 kinds give. A pre-arm confirmation
  or invalidation comes back from `chart_watch.evaluate_h1_bars` as `pre_arm` - a
  `chart_watch`-level verdict, **never an indicator reason**, the frozen rule sheet is not
  edited and not asked a different question: while a pre-arm closing-through bar sits
  inside the rule's age window the rule keeps saying `invalidated` and the watch simply
  waits, exactly as it waits on `awaiting_reclaim`. Nothing fires, nothing is recorded, no
  push, no feed row, and the watch stays armed. The comparison ATTACHES the desk's
  market-local zone to a naive stamp and keeps an aware one as the instant it is
  (`autopilot_core._gate_moment`'s pattern, never `chart_watch._naive`): stripping an arm
  written three hours west of the desk would read it three hours EARLIER than it happened
  and turn a pre-arm arm back into a post-arm one. `armed_at` round-trips through
  `chart_watches.json` unchanged, so a restart is not a second chance, and a disarm +
  re-arm is a new `watch_id` with a new `armed_at` to which the bounce that already fired
  is pre-arm too.
- **Where it fires.** The armed poll is `_poll_d1_event_watches`, and the H1 pass runs at
  its HEAD - before its "no D1 event watches armed" early return, which would otherwise
  make the feature invisible whenever the trader had no D1 event armed. The fired event
  lands on the **D1 feed as an ARMED event** (`CHART_WATCH_TAG`, timeframe D1), never as
  a detector alert and never on the M5 alert list. One event per watch, carrying **every**
  measured reason, then it disarms; re-arming is a new `watch_id`.
- **The phone.** `PriceAlertService.notify_armed_watch` - the SAME sender the armed
  Research/Focus price alerts use, in every mode, de-duplicated by watch id. The
  exception was never about price alerts; it is about a condition the trader armed by
  hand (`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`, amendment 2026-09-13). The push goes
  out BEFORE the alert is drawn: a broken display path must not be able to suppress it.
  **DISPATCHED before the alert is drawn, DELIVERED on the service's own worker**
  (repair 2026-09-13, review blocker B3): the send itself is `push_notify`'s 10-second
  HTTP call and the caller is the GUI poll, so on the Qt thread `notify_armed_watch` now
  makes only the cheap decisions - the engine check and the watch-id de-duplication,
  where the id joins `_announced_watch_ids` BEFORE the dispatch so a repeat in the same
  tick is refused without waiting for the first send - hands the send to a one-shot
  daemon thread the service owns and tracks (the `check_now` pattern; an armed watch
  fires once and disarms, so a standing consumer thread would idle for days to serve a
  handful of sends), and returns `{"ok": True, "queued": True, "watch_id": ...}`. **`ok`
  means "accepted for delivery by the one armed sender"**, not "a push left the desk";
  the outcome arrives afterwards on `_last_push_error`, the `ARMED WATCH ...` log line
  and `statusChanged`, exactly as `_notify` already reports one, and a transport that
  raises is logged with its traceback on the worker rather than lost. `shutdown()` joins
  what is in flight with ONE budget, `ARMED_PUSH_SHUTDOWN_WAIT_SECONDS = 2.0`, then
  returns - the threads are daemons, so a dead endpoint can never hold the process - and
  because a worker's `statusChanged` is QUEUED to the GUI thread by Qt, the thread that
  waited emits the final snapshot itself (the event loop it would have been delivered on
  is the one that just stopped). No `auto_mode` gate is added: DESK, AWAY, EVENING and
  OFF all still deliver.
- **Cost on the Qt thread.** Per ARMED H1 watch, per 60 s tick: one O(bars) bucketing
  pass over M5 dicts `_m5_bars_for` has already materialised, then an O(bars) ATR and EMA
  over the ~35 resulting H1 bars. No fetch, no file read, no second copy of the series.
- **The open limit, stated rather than hidden.** The desk's cached M5 window is five
  sessions at `useRTH=1` (SN2's `"5 D"`), which aggregates to about **35** completed H1
  bars against a **45**-bar warm-up. Two sources were checked before settling for it
  (lead ruling 2026-09-13): WS-CH's *Load older* path is the SAME `m5_chart_bars` call
  with a ceiling of ten sessions on the ASK, and one `"5 D"` window behind it; and the
  durable H1 store `project_paths.MASTER_AVWAP_INTRADAY_BARS_DIR` **does not exist on
  the live desk** - nothing has ever written it. So the watch waits and SAYS SO: `not
  measured (N of 45 H1 bars)`, counted from the bars in hand rather than from a
  remembered number. An armed surface reporting `ok` beside a watch that cannot
  evaluate is the one thing that table must never say.
- **So the history is fetched, for armed symbols only** (lead decision 2026-09-13; the
  trader may overrule, and a watch that cannot fire for a whole test week gives them
  nothing to judge). `scripts/h1_history.py` reads ONE armed symbol's hourly bars
  through `yfinance` on its own daemon thread - the group RS/RW tape precedent: its own
  clock, zero IB traffic, no `bounce_bot_lib` change. The cache stays PRIMARY (a full
  window never touches the network, and a test holds that), at most one fetch per
  completed H1 bar per symbol, never on the Qt thread, completed bars only through
  `completed_bars.is_completed_bar`, and the exchange zone CONVERTED with `astimezone`
  before it is dropped - N1's fault, not repeated. A failed download is a REFUSAL:
  `not measured (N of 45 H1 bars, yfinance unavailable)`, never an empty tape. The
  health cell names the source, `H1 from cache` / `H1 from yfinance`, so no verdict is
  read without knowing which history produced it. The alternative - a wider M5 window
  for armed symbols in `bounce_bot_lib` - stays the trader's ask; `v1` keeps its 45.
- **The need is measured on the PRIMARY series, never on the one that was chosen**
  (repair 2026-09-13, review blocker B1). `_h1_bars_for_watch` asked for a refresh only
  when the CHOSEN series was short of the warm-up, so the moment the fallback held 45
  bars nothing ever asked again while the desk's own window stayed at ~35 for ever - the
  watch was then judged on ageing bars until the rule's 24 h staleness answered "not
  measured" for good. `chart_watch.h1_bars_for_watch` returns the yfinance series only
  when the primary is short, so that source IS the short answer and the panel asks on it.
- **The refresh cadence is a completed SESSION-ALIGNED bucket, not the wall-clock hour**
  (repair 2026-09-13). `H1HistoryCache.request` keys its refusal on
  `h1_history.last_completed_h1_bucket` - open-relative buckets 06:30, 07:30 ... 12:30
  market-local, the last one 30 minutes long and closed at the bell - because a new
  answer can only exist when one of those has completed. The clock-hour key both
  refetched twice inside one bucket (11:45 and 12:15) and refetched every hour all
  evening, when no bucket can complete at all. A fetched bar is admitted on the same
  rule: `h1_bucket_end` walks the short 12:30 bucket to the 13:00 bell, so the two
  series agree instead of standing one bar apart for an hour every session.
- **A refresh that fails after a success keeps the bars and says they STOPPED** (repair
  2026-09-13). The held bars are still the best answer available and the watch keeps
  being judged on them, so the health cell reads `H1 from yfinance (stale - last refresh
  failed)` through the new `H1HistoryCache.last_refresh_failed`; the next completed
  bucket retries. `unavailable` keeps its own meaning - nothing was EVER fetched -
  and still prints `not measured (N of 45 H1 bars, yfinance unavailable)`.
- **Tests:** `tests/test_ws_10c_h1_retester.py` (the packet's, written red - 25 of 26
  green; the 26th clicks an `ArmBar` with no symbol charted, where every watch toggle is
  disabled by design, and is left red rather than weakened) and
  `tests/test_ws_10c_h1_retester_builder.py` (11 added, including the M5-poll date-roll
  case proven to fail with the `watch_is_stale` exemption removed).
## 10I - two contexts per row, three verdicts per thesis (2026-09-13, WISHLIST 10I/10K)

### The question, and why it needed a contract rather than a column

The trader keeps two accounts of every decision: what they EXPECTED (the thesis) and what
the tape was MEASURED to be doing (the environment). WISHLIST 10I asked for those two to
be connected to the opportunities and the trades without either one contaminating the
other - "one drillable answer with three populations" - and 10K asked for the environment
to be measured first and connected afterwards.

The connection is where the errors live, not the measurement. Three of them are the reason
this is a module and not a join:

* **A label is not available when the session it is about starts.** `d1_environment_v1`
  reads a session's completed daily bars, so the label for 2026-09-08 exists at 2026-09-08
  16:00 ET. A 10:35 decision on that session cannot have known it. Handing it that label
  reads as evidence and is a leak: every cut built on it would be scored with information
  from after the decision.
* **One calendar day back is not the previous session.** The fixtures are built around
  Monday 2026-09-07, Labor Day, precisely because a `- timedelta(days=1)` walk from
  2026-09-08 lands on a day nobody labelled and turns a measured cell into `unknown`. The
  walk is `market_calendar.previous_session`.
* **15:35 ET is 19:35 UTC.** A read that strips the offset decides the close has passed
  and hands out the session's own label. Both stamps go through
  `journal_trade_shape._coerce_datetime` - naive ATTACHED, aware CONVERTED - which is the
  journal's own rule and not a second opinion about the trader's clock.

### The time rule, stated once

`scripts/context_join.py` owns it and nothing else restates it:

| the row's clock | session read | certainty | flagged |
|---|---|---|---|
| before that session's 16:00 close | the PREVIOUS exchange session | `prior_session` | no |
| at or after the close | its own session | `session` | no |
| an OBSERVATION dated by session alone (`scan_date`) | that session | `session` | no |
| no time of day - a bare date on an ENTRY, or midnight market-local | the previous completed session | `date_only` | YES |
| a session nobody labelled | none | `unknown` | YES |

The fourth row is the broker-file case: `journal_statement_import` stamps every fill
midnight market-local precisely so `journal_trade_shape.is_date_only` can recognise it.
**A date-only fill may never be given a midday regime** - that would be the desk claiming
it knows the fill happened after the close.

The text form is what separates row 3 from row 4: `2026-09-08` and `2026-09-08 00:00:00`
both coerce to midnight, and they are different statements. A scan row's date NAMES the
session it read completed bars for; a fill's timestamp is a MOMENT, and midnight is the
one time of day a fill cannot happen at.

### Two refs, never one

`observation_context` and `entry_context` are separate columns and neither overwrites the
other. An opportunity nobody took has the first and not the second. A trade has both and
they routinely disagree - that disagreement is the readable part, and blending them would
destroy the only thing the pair is for. `ContextRef` carries `context_id` (benchmark, rule
version and the session the label is ABOUT), `observed_at`, `available_at` and the
certainty. **The certainty belongs to the JOIN, not to the context**, so it is not part of
the id: a live read and a reconstructed read of the same reading are the same context,
told apart by the certainty and never by the label.

### A thesis links by scope and window, and none is chosen

`link_theses` links when the row's benchmark scope and the thesis's validity window cover
the row's observation time. The scope comes from the row's own `ContextRef` - which is why
`attach_context` runs FIRST - because the tier-outcome rows carry no benchmark column and
inferring one from the symbol would link a QQQ call to an SPY decision. The window is
`created_at` to the horizon's last session close (counted in SESSIONS) or `invalidated_at`,
whichever comes first: a thesis written on Thursday is not evidence about a Tuesday
decision, and an invalidated thesis stops covering what comes after it while keeping what
came before - it WAS the stance at the time.

Overlapping theses ALL link, newest first, and the payload carries no `chosen`, `primary`,
`selected`, `best` or `winner` key. Ambiguous stays ambiguous; the review grades each.

### Three verdicts, three keys

`setup_environment_evidence.thesis_review` answers three different questions and never
merges them: `market_call` (the benchmark's own later path against the stated stance),
`setup_held` (the linked opportunities' record) and `trade_profitable` (money, once per
trade). The case that makes the rule is the common one - **the call was RIGHT and the
trade LOST**. A single grade over those two facts is worth less than either.

`market_call` is `open` whenever the recorded path has no close at the horizon's last
session. Not reached, not recorded and not directional are all `open`: a thesis is never
called wrong because the data stopped.

### What the cells refuse

`opportunity_cells` reports every cell and lets some of them lead. A cell under
`evidence_stats.MIN_REPORTABLE_N`, or with more than `working_lately.CONCENTRATION_LIMIT`
of its sample from one name or one session, prints with its refusal in `lead_refusal`. On
the packet's fixture the best-looking rate on the page - 85% - is thirty rows of one
ticker, and it is refused; a cell split evenly over two sessions sits at exactly 0.50 and
stays eligible, because the limit is EXCEEDED and never reached. A `win` column that is
present and EMPTY is unmeasured, not a loss.

`personal_cells` keeps the grains apart: one trade with two confirmed tags shows in two
cells and contributes its money ONCE (`duplicate_tag_rows` is the difference, the rule
`preference_trade_outcomes.trade_level_summary` already owns). Theta, day trades and
unconfirmed tags are excluded BY NAME with three different reasons and counted in their own
populations - a sold put is premium, not a swing. An open position's mark is never money.
A commission keeps the sign the importer gave it.

### The day-trade cut reads a different word

The M5 outcome rows already carry `context_json.market_environment`, stamped at alert
registration and read by `held_run_score` and the digest's `env_key_of`. **That is not the
D1 label.** It is a different vocabulary on a different clock answering a different
question, and the 10I day-trade cut is the D1 label of the row's `trade_date`. Printing
the two under one heading would be two unrelated words in one column.

### The surface

Research > Results gains a page-level **By environment** control (`ENVIRONMENT_ALL` is the
default and cuts nothing, so every champion golden is byte-identical). Bot is cut by the
environment known at OBSERVATION and My trades by the one known at ENTRY, and the control's
line names the benchmark and the rule version, because `compressed` is a reading under a
versioned rule and not a word about the weather. Bot x Day gets its own section key and its
own statistic; a day cell is never a row inside the swing cut. Two populations, two readers,
both on the Results worker: the day page streams the intraday outcome log and never opens
the swing tier file.

### Backfill

`python -m context_join backfill --since YYYY-MM-DD` is DRY BY DEFAULT, names the data dir
and `--apply`, and labels only what the contemporaneous store establishes. Every
reconstructed ref is `certainty=reconstructed` and flagged, and is kept out of every
forward claim: the label is the same label, and what makes it different is that nobody
read it at the time.

**Tests:** `tests/test_ws_10i_context_join.py` (the tester's 23 plus a fixture-calendar
guard, red before any of this existed at `6e3854aa`) and
`tests/test_ws_10i_context_join_build.py` (the builder's 9 for the surface seams the
tester's file does not pin, proven red by restoring the pre-change files).

### The Daily Recap's two labels

WS-DR landed on the sweep branch the same day, so the wiring is here rather than owed.
`RecapRow` keeps its own `d1_environment` - the label OF the session, which is what WS-ENV
joined and what WS-DR's tests pin - and gains `observation_context` beside it, which is
what the decision COULD KNOW. For an intraday row those two differ by one session and the
difference is the point: a 10:35 alert is labelled with the previous session's tape, and
the Environment cell's tooltip still names the session's own label so nothing is hidden. A
matched decision carries `entry_context` from the preference report's `trade_opened_at`,
and the cell prints `observed → entered`; an unmatched opportunity prints one label,
because there is no fill and there is nothing to invent.

**Reopen trigger.** A second benchmark is cut on; the trader asks for the thesis review on
a screen of its own; a recap row's own `d1_environment` is reconciled with the observation
context (a lead decision, not a builder's).
## The sentences CLAUDE.md moved here on 2026-09-13 (WISHLIST sweep docs pass)

`CLAUDE.md` had grown to 52.0 KB against its ~45 KB limit (the 0.25 memory section and the sweep's rule lines). The 34 longest bullets were shortened to their seams and each one's ORIGINAL text is reproduced below, verbatim and unedited, in `CLAUDE.md` order. **Nothing was deleted.** Where a bullet below and the current `CLAUDE.md` differ, the shortened rule in `CLAUDE.md` is the binding one and this is the full account behind it; the incident sections elsewhere in this file remain the deeper record. The one wording change beyond shortening: the auto-tagging rule now names FOUR lanes (WS-10E's `trader_note` lane), the story under "The four auto-tagging lanes".

- The post-scan build runs in an owned CHILD PROCESS at below-normal priority (F1, BD-95), never a thread — a CPU-bound thread holds the GIL and no priority trick frees the GUI. Reads are session-scoped; partitions are MONTH-keyed, narrowed through `ResearchStore.read_rows` (`symbols` / `interval_start_range`), never by filtering a materialised list. Growth resets on the 1st.

- The seal de-duplicates at the dataset grain and counts what it drops (`SealResult.rows_deduplicated`; `SUPERSEDING_DATASETS` exempt). Repair is `research_warehouse.cli dedupe --apply` (dry run by default); derived bars and intraday features computed from a duplicated month are wrong in VALUE and need a rebuild, not a dedupe (BD-96/97).

- H2/H4 exist for the HTF LRSI study (BD-78) and end each session with a STUB, published and excluded from the LRSI input. The HTF LRSI grid is 16 diagnostic recipes (`outcomes.HTF_LRSI_RECIPES`), never a Cartesian search; both legs read the same unmirrored series (BD-79). Live `CROSS_LEVELS` stays `(20, 50)`.

- `anchor_instance` comes from `earnings_avwap_anchors.csv`, which the SCAN feeds through `runner.bridge_earnings_anchor_caches_to_csv` → `append_anchor_candidates` (append-only, de-duplicated on ticker + anchor_date, new rows at the END, failure logged never raised). Nothing live reads the CSV; never trim it — the newest two dates per ticker are the current and previous anchors.

- A reconstructed anchor is LABELLED and never promotion evidence (BD-99/100): `AnchorChoice` is `observed` / `reconstructed`, `feature_snapshot_daily.anchor_knowledge` carries it (NULL reads `legacy`), `outcome_path.path_kind` names the swing path and is excluded from BD-98's unchanged-comparison. Repair order: build → `rebuild-daily-features` (dry run by default) → `recompute-outcomes` → `band-coverage`. A terminal outcome row is re-simulated only with `force`.

- **The daily snapshot carries BOTH AVWAP band families and they never share a column** (packet M4, 2026-09-05, BD-102). `avwape_*` is the champion (frozen, decision 0008); `avwap_variant_*` + `avwap_variant_formula_version` is the challenger, computed by `indicators.avwap_band_variants` from the SAME bars and anchor index and **independently of whether the champion produced bands** - a NULL band is "not measured", never a band on the centre line. `swing_house_variant_v1` differs from `swing_house_v1` ONLY in `band_family`, its `outcome_definition_id` (`band_variant_v1`) fences it out of every `house_default_v1` reader, and `band-coverage --compare A B` pairs the two on the SAME occurrence ids. Shadow only; T4's criteria decide and nothing here promotes. The version rules, the recipe's band map and the unpaired-occurrence count: DESK_INTERNALS "M4 - both AVWAP band families".

- The setup registry (`scripts/setup_registry.py`, `setup_registry_v1.json`) is frozen DATA, regenerated by `build_setup_registry.py --write` with the diff reviewed, resolves no disagreement and fills no column its sources do not establish; an unknown name RAISES. It becomes authoritative at `plan.md P4.1`; nothing in production imports it yet. `trial_ledger.register` writes one append-only row per grid BEFORE any outcome is read and refuses to rewrite a `trial_id`.

- **The `setup_research` narration is a BOUNDED view and its selection is a SIZE rule, never a ranking by result** (N3, 2026-09-05, BD-101). `_bounded_narration_view` encodes the fixed head first, then eligible policy cells by **`stats.n` descending, then `recipe_id`/`family`/`side`** until the next would cross the budget, then the after-like ELIGIBLE cells by their top-level `n_episodes`. **No `mean_r`, `win_rate`, `profit_factor`, `expectancy` or any R statistic may enter that key** - gate #43 is a refusal - and `narrated K of N` is stated in the view, the `.narration.json`, the pack markdown and the ledger reason. The numbers behind it: DESK_INTERNALS "N3 - the bounded narration view".

- **Entry comparisons keep the opportunity denominator and the fill denominator apart** (Phase 0.32 Packet 2, 2026-09-15). `entry_comparison` is a caller-supplied-row reader: its P8 adapter emits every declared entry variant for every base opportunity, so a no-trigger or unavailable measurement is visible rather than silently disappearing; its old-M5 adapter retains a missing-data base attempt and does not turn exit recipes into entry samples. A shared-triggered gross-MFE delta and all-opportunity coverage answer different questions and are printed separately. Scanner, like, veto, actual-trade and unreviewed rows never pool; repeated scans collapse at the attempt seam while distributions and pairs use one stable `dependency_cluster_id` representative. A review comparison refuses when knowledge basis/timing/window, coverage or entry convention differ; normalized rows carry Packet 1's entry rule/version into that convention. Every public adapter resolves supplied recipes through the existing ledger and rejects a forward variant outside its declared entry axis; it reads ownership and lifetime looks but cannot write, register, amend or launch a trial. Under floor or before maturity it says `not_evaluated`, never winner. This remains pure research with no store or Qt work and no live influence.

- **A VETO retires the chart, a CLAIMED like ADVANCES it, a QUICK like and a NOTE move nothing** (trader, 2026-09-04). A rail veto uses its own verb (`vetoRetireRequested` -> `_retire_after_veto`) and writes ONE row; a quick like is `likeRecorded` -> `_after_like` and a claimed like `likeAdvanceRequested` -> `_advance_after_like` -> `_advance_review_queue`, which parks nothing and drops nothing. Both write `like_advance` through `_record_like_advance` because `review_learning.TAKE_ACTIONS` keys on it. The note-box rules and the "Veto D1 - but M5 today" ordering: DESK_INTERNALS "T1 - the four capture verbs".

- **A day-trade PASS is a note, not a veto, and never retires the chart.** Its multi-select codes are a SEPARATE vocabulary family (`ui/annotations/vocabularies/pass_reasons_v*.json`), written in vocabulary order; cached M5 bars are referenced through a sidecar written BEFORE the row (`ui/annotations/pass_bars.py`); a capture click never fetches; a pass does not mark the symbol "Reviewed today" (`pick_feedback._ANNOTATION_DECISIONS` stays `veto`/`like_claim`/`note`).

- **A LIKE has two modes and only one names a setup** (P9). **Alt+L** is the QUICK like (`like_mode: "quick"`, no claim, no why, never prompts) and **Alt+K** is the CLAIMED like; the quick-like BUTTONS prompt for an optional note, the key never does, and an absent `like_mode` reads `claimed`. A like carries zero privileges, grades under `like_unclaimed` when quick, and contributes a LINK to the auto-tagger, never a tag. `sidecar_completion` finishes a capture sidecar into a NEW file (`m5_bars_completed_ref`, `m5_bars_ref` never rewritten) and **that read is AWARE** (N1, 2026-09-05): `pass_bars.desk_zone()` is ATTACHED to a naive `dt`, never an offset stripped. The fault names and the incident: DESK_INTERNALS "P9 / N1 - the two like modes and the aware sidecar read".

- **Every verdict has a forward record and no two verdicts are combined** (P5): veto, like, pass, rejection (`focus__m5_not_today` / `focus__swing_dislike`, the double underscore load-bearing). The rejection family's pooled base row is LABELLED and never read as either verdict. Pass code cohorts overlap and are never summed; only `pass_all` counts passes. `unfavorite` is never graded; a rejection's free-text reason is never machine-coded. A pass grade the sidecar cannot reach is BLANK with `intraday_unmeasured_reason`, never zero. `update_human_focus_outcomes`'s `pick_key` defaults to the existing identity.

- Intraday alerts are a list beside the chart (`ui/widgets/m5_alert_bar.py`, LEFT column), not a queue in front of it. Clicking from one row to the next is a SKIP, never a re-queue; **a click away IS a pass** (trader, 2026-09-01) — never "fix" it, and never rename `clicked_away_from_m5_alert`. Routing is `_is_m5_review_alert` inside `_enqueue_review_alert`, after the AWAY branch.

- **A quiet Focus pick FADES after 10 trading days**, reversibly (`focus_pick_clocks.json`, reset by a fired flag, a watch hit or "★ keep"); applies to the trader's own picks too, routed through the store's own removal path so a watchlist line is never touched. Faded is not deleted (`focus_faded.json` + append-only row); a faded swing favorite appends a RETRACTION; no `pick_feedback` verdict is written for a fade. `FocusPickStore` is the single writer; the check runs on the day roll and a half-hourly timer, never in the 60 s poll.

- Today's swing picks (`ui/widgets/swing_favorites_bar.py`) get two writes — swing Focus FIRST and must not fail, then the append-only `swing_favorites.jsonl` row whose failure is swallowed. Never write an auto-adoption marker for one; like-origin is `vetted` (cohort `human_focus_swing_vetted`); a removal appends a RETRACTION. The "taken" badge is a display-only join off the Qt thread. The strip was the bottom of the M5 alerts column until 2026-09-14; it now sits UNDER the setups in the right (D1) column behind its own draggable split ("D1C-L - M5 left, D1 right").

- The review scoreboard grades every explicit decision; an action joins `TAKE_ACTIONS`/`REJECT_ACTIONS` on what its WRITER does, never on its name (`veto_day_trade` is a REJECT). Machine events, `*_fired`, `*_expired` and `disarm_*` stay out. `r_gap` is report-only, fires on the R difference alone, and is absent from `draft_policy_from_state`, `review_guidance` and the AI evidence package. Coded vetoes annotate `dislike_reason` and never re-resolve an episode.

- A sweep-finalized trade counts under the policy that MEASURED it: `setup_scoreboard.exit_policy_r` keeps `eod_hold` / `stop_exit` / `last_measured` as separate columns, never blended, and every eod-hold view reads `r_eod_hold`. **`unresolved` means UNMEASURED** (M2, 2026-09-05): a swept trade that measured its bars is written `swept_measured`, and every reader goes through `outcome_semantics.terminal_kind`. The champion's eod-hold tier cells still take `eod_complete` only, BY DECISION (2026-09-06) and pinned by a golden characterization in `tests/test_setup_scoreboard.py`; the reasoning: DESK_INTERNALS "M2 - a swept trade counts under the policy that measured it".

- Auto-tagging has three lanes that never compete and are ordered by LANE, never confidence: `journal_analytics.AutoTagger` (which setup), `journal_trade_shape` (facts from the trade's own timestamps and legs), and `trader_capture` (what the trader already SAID inside the trade's own window, outranking every fuzzy source). No tag is ever derived from the outcome and unmeasurable emits NO tag. The rejection prefixes, `context_row_id` and the match-confidence rule: DESK_INTERNALS "The three auto-tagging lanes".

- The trader owns `trade_annotations`; the ONE machine writer is `scripts/journal_bulk_tag.py`, writing only `tag_status='provisional'` for a CLOSED trade with no confirmed tag (the refusal to overwrite lives in `JournalStore.apply_provisional_tags`), never a shape tag, never `tag_corrections`; below threshold only a `needs_review` marker. "My setups" counts confirmed tags only.

- **`preference_trade_outcomes` matches inside 10 SESSIONS** (`statement_window_end`; a calendar refusal falls back NARROWER) **and `trade_level_summary` sums P&L once per `trade_id`**. **`journal_exposure` reads bias from the legs - a LONG option is never a bullish setup**, and a `trade_legs` row is a FILL. **`personal_evidence_summary` partitions by STATUS** - complete / partly_closed / open_exposure - **with `uncertain` a CROSS-CUTTING label that pools no money**, and names no best setup below `MIN_REPORTABLE_N`. The spread seam, the one denominator and Weekend Prep's backlog rule: DESK_INTERNALS "The personal-evidence reader rules".

- **A broker file is authoritative for money and blind to time.** `journal_statement_import` (Questrade `.xlsx`, no `openpyxl`) and `journal_ib_transactions` (IBKR sectioned csv, USD price / CAD money) write executions at MIDNIGHT market-local, and `journal_trade_shape.is_date_only` refuses to name a session for them. Identity is `fill_signature` plus an ordinal, never positional, and **commission carries a SIGN the importer owns - nothing downstream may `abs()` it**. The account unmasking, the description parse and the richer-source rule: DESK_INTERNALS "A broker file is authoritative for money".

- **The tracker replay has a versioned execution convention and level knowledge** (ST3/ST7, `master_avwap_lib/execution_convention.py`): default since 2026-09-06 is `gap_aware_v2` (gapped fills at the OPEN, no open clamps into the bar, an invalid candle books nothing and defers the max-hold `TIME_STOP`) / `prior_session_v2` (only INTRABAR target tests move); `literal_level_v1` / `same_session_v1` remain by name, both stamped.

- **Which observation of a thesis gets graded is a NAMED policy** (ST4/ST7): family rows carry `selection_policy`, `first_actionable_v2` (`master_avwap_lib/selection_policy.py`) is the default since 2026-09-06 (decision 0019), `closed_first_v1` remains by name, and the persisted write logs `policies: selection=.. execution=.. levels=..`. A scenario's exit date is its last `events` entry, read only when CLOSED, and a COMPACTED record is `undatable_exit`, never silently pending. **A compact projection's `_scoring_outcome_summary` IS the record** - recomputing dropped every setup and zeroed both deltas: DESK_INTERNALS "ST4 / ST7 - the named selection policy".

- **The AVWAP band challenger is measured through the CATCH-UP path too** (M1, 2026-09-05): `build_anchor_band_variant_meta` lives in `legacy.py` and serves BOTH the live scan (`runner.py` re-exports it) and the tracker catch-up (`_evaluate_priority_snapshot_for_date`). Never add a third builder; a record is rebuilt on every persisted tracker write, so no migration exists. The Band variant tab prints its coverage from the EXPORT's own counts, never by reading the 1.1 GB tracker; the ten silent days and T4's accrual: DESK_INTERNALS "M1 - a shadow that measured nothing for ten days".

- **The control, study and experimental-exit populations are SURFACED, LABELLED, and never mixed with picks** (M5, 2026-09-05): three Setup Tracker tabs - Controls, Studies, Exit frameworks - read three CSVs written in the tracker's own guarded save pass. Win rate leads with `n` and the ONE Wilson bound, the sort is the bound, each tab carries a population sentence and `experimental` is a COLUMN. Shadow only; the champion aggregates are pinned byte-identical. The exact sentences and the two reconciled n's: DESK_INTERNALS "M5 - the control, study and exit-framework populations".

- The digest gate has TWO halves (Q4): `clean_digest_sessions` counts CONSECUTIVE clean exchange sessions walked through `market_calendar.previous_session` (clean = `is_session` plus an EMPTY `unavailable`), and `digest_audit_approval.json` is written **only** by `python -m ai_jobs.digest approve-audit` (from `scripts/`), never by a nightly job. `gate_met = window_met and audit_recorded`; `journal_enrichment` refuses until both are true; the System Health strip shows `sessions_consecutive_clean`.

- **M5 Strength Board:** batched yfinance over `universe_all.txt` PLUS the four trader watchlists, zero IB traffic; relative volume is SESSION-RELATIVE and is not one of the seven fenced formula functions (byte-identical to the R8 baseline). Its parity rows auto-join M5 Focus through `_auto_adopt_strength_board` - DESK only, the ONE adoption gate re-run per row, never removing - and every Focus add is injected into `longs.txt` / `shorts.txt` by `FocusPickStore._inject_into_shared`, a removal un-injecting it. The skip list, the batching and the SMA floor read: DESK_INTERNALS "The M5 Strength Board's auto-adoption".

- **The Desk's Strength window is ONE flat scrolling page** (`ui/widgets/strength_page.py`, trader 2026-09-07): Focus strength, the auto RS/RW board, the RRS snapshot with its three scopes STACKED, then the M5 Strength Board under a page-owned heading; every text board is sized to its document and every table to its rows (capped at `FIT_ROWS_CAP`), so nothing inside the page scrolls on its own. No tabs, no collapsible sections, one always-on scrollbar, and the page's 170 px floor keeps the alert column's 360 px budget. One `StrengthBoardService` owned by `MainWindow`. Story: DESK_INTERNALS "The Strength window is one flat page".

- **The priority switch reorders and never withholds, and it is BUILT** (ST6, 2026-09-06): `local_settings` key `prioritise_working_lately`, default OFF, read AT SORT TIME and never at write time, stably re-ordering the M5 list, the WAITING review list (where the panel picks the next chart) and the setups table by the snapshot's verdict order with ties keeping arrival order; the tier gate, movers-only and the repetition fold are untouched, and the identical-visible-rows test exists.

- **ONE Working-lately snapshot, four surfaces, and nothing called proven** (ST6): `working_lately.build_snapshot` is PURE and `snapshot_id` is a sha1 over the sorted cells + the declared policy lines + `as_of` alone, so a timer tick cannot move it; `ui/services/working_lately_service.py` owns the worker build, `snapshot_latest.json` and the deduplicated `leader_change_events.jsonl`, and four surfaces render THAT payload. Dependence is answered by REFUSING - a cell over `CONCENTRATION_LIMIT` (0.5) of one name or session cannot lead, `pool_cells` RAISES across kind/side/outcome kind, and every surface prints `observational leader among K cells`. The cause precedence, the persistence rule and the labelled `panel read`: DESK_INTERNALS "ST6 - the one Working-lately snapshot".

- **Win rate leads every trader-facing SWING surface** (first, with `n` and a Wilson lower bound from `swing_headline`, sorting by the bound, mean R beside it); **MFE-after-a-held-level leads every DAY-TRADE surface**. ONE WILSON: `swing_headline`'s z (1.96); `expected_r.py`'s 1.28 is a parameter inside a fenced file. **Counts are integers at each table's own grain** (ST2), **there is ONE leader through `working_lately.select_leader`** (four states, computed once per page by `panel_verdicts`, a study never leading), and **the tier outcomes `win` is a FAVORABLE-DIRECTION flag at a scan-row offset, not a stop-rule win** (ST1, `swing_evidence.read_eligible_rows` the ONE reader). The exported count names, the declared margins and the v2 shadow file: DESK_INTERNALS "Headline statistics - the ST1/ST2 clauses".

- **The day-trade headline is `held_run_score`**: P(held in the first 30 min) x trimmed-mean MFE_R of the held ones, ONE formula reaching every surface, segments spelled the AGGREGATOR's way. **Held is MEASURED held** (Q1): `measured_held` / `measured_broken` / `pending` / `unmeasured`, `hold_rate` = held / MEASURED, the unmeasured counted and shown, never assumed. My Decisions rows use `ALL_DIRECTIONS`, a pooled cell accumulated from the episodes and never an average of two cells, and the window is `evidence_stats.lately_window` with gaps reported. The two join seams, `UNDERIVED_DIMENSIONS` and the D1 alignment: DESK_INTERNALS "Q1 - the day-trade headline and measured held".

- **The AWAY digest ranks swing picks by the tracker's record**: Wilson lower bound on the family's realized win rate from `master_avwap_tier_outcomes.csv` inside `lately_window()` at ONE declared horizon (`evidence_stats.SWING_HORIZON_SESSIONS`, 5), expected R as tiebreak, ungraded families below every graded one, `stale_horizon` rows dropped, the near cap applied after ranking; the bucket is printed, never ranked on.

- **Research is not a trader surface - except its Results page** (decision 0016 answer 7, amended 2026-09-06; `scripts/research_results.py`): the readable full readout, four populations never pooled over ONE ST6 snapshot, computing no new statistic. Its window control APPLIES to My trades (`in_window`, on `closed_at`) and is DISABLED on Bot, whose cells carry their own `window_sessions` — a control that only labels is a false claim (`docs/DESK_INTERNALS.md` "G5"). Nothing the trader must see may live only in the other eight tabs.

### Also moved on 2026-09-13: the memory, frozen-exe and where-to-read-more bullets

Shortened in the same pass; originals verbatim. The runbook list is now `docs/README.md`'s.

- **`MEMORY.md` at the root is a ROUTING INDEX only** (name -> detail file -> trigger
  keywords, never a fact); the detail lives under `memory/` (`people/`, `projects/`,
  `decisions/`, dated daily notes, prunable `context/`). At idle boot read the standing
  instructions and `MEMORY.md`; once a task exists the narrow reads below apply unchanged.

- **Before answering about prior work, decisions, dates, people or preferences, search
  memory first:** route through `MEMORY.md`, read the narrowest matching detail file, use
  at most five sources for the recall answer (this never caps the reading a build or audit
  needs), and cite file, tag and date. Unverified freshness is stale; a live-status
  question is answered from the checkpoint and the code, never from memory.

- **Memory is recall, never authority.** `CURRENT_CHECKPOINT.md` is the brief, `plan.md`
  the build order, `CHANGELOG.md` the inventory, `docs/decisions/` the contracts and the
  code the fact. A detail file outranks its index (repair the index); nothing in memory
  outranks those. No recalled line, inferred lesson or signal count authorizes a detector
  change, overrides an accepted decision, promotes a WISHLIST item or bypasses the
  ask-first rule.

- **Every non-blank line in `people/`, `projects/` and `decisions/` carries `[stated]` /
  `[observed]` / `[inferred]` / `[suggested]`, a date and a source**; only the trader's own
  words support `[stated]`. Keep only what cannot be re-derived from git, the control set
  or a live store; never keys, account numbers, live counts or machine status. An inferred
  lesson becomes a standing rule only after three weighted independent signals across two
  sessions (a signal older than 30 days counts half); a trader correction applies at once;
  a failure memory says what broke and what fixed it and is never an order.

- **Supersede in place** (strike the old line with its date, the tagged replacement beside
  it); update `MEMORY.md` in the same commit as the detail; consolidate before 15,000
  characters per file. Recon, reviewer and tester never write memory - they hand the lead a
  proposed sourced line; a builder records only an in-scope durable detail; the lead
  integrates. Claude's machine-local auto-memory is private scratch, not the shared record.

- Active specs: the Phase 0.5 packets R1–R8 (`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`, `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md`, `docs/SWING_QUALITY_AND_FEEDBACK_PLAN.md`, `docs/DESK_CHART_UNIFICATION_PLAN.md`, `docs/M5_SIGNAL_ENGINES_PLAN.md`, `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`, `docs/WEEKEND_PREP_PLAN.md`), the warehouse trio (`docs/ULTIMATE_SETUP_DATABASE_PLAN.md`, `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`, `docs/RESEARCH_WAREHOUSE_ERD.md`), `docs/LOCAL_AI_AUTOMATION_PLAN.md`, `docs/REVIEW_LEARNING_LOOP.md`, `docs/CHART_REVIEW_WORKSPACE_PLAN.md`, `docs/AVWAP_BAND_VARIANT_STUDY.md`, `docs/DURABILITY_CATCHUP_PLAN.md`, `docs/SETUPS_MAJOR.md` / `docs/SETUPS_TEST.md`.

- Runbooks: `docs/FIRST_SESSION_CHECKLIST.md`, `docs/AWAY_SCANNER_RUNBOOK.md`, `docs/EVENING_MODE_RUNBOOK.md`, `docs/REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md`, `docs/GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md`, `docs/MACOS_SETUP.md`, `docs/SHIP_READINESS.md`, `docs/BROKER_ADAPTERS.md`, `packaging/README.md`.

- Runtime facts: the main desk is an always-on Ryzen 7 8845HS mini-PC (32 GB, Radeon 780M iGPU — local-LLM host) and does everything; the old i5/3080 Ti desktop is powered down most days and never a writer. Storage is a DAS at `\\MINI-PC\Trading Bot Data` holding `research_lake/`, `ai_store/` and the cold-pushed subtrees. A full scan measured 17–21 min over 1,097 symbols on the 8845HS. Post-session artifacts under `%LOCALAPPDATA%\TradingBotV3\diagnostics\`.

- **Do NOT rebuild per commit.** Rebuild before each merge to `main` and immediately when a change hits a trigger below; ask before spending the trader's time on the click-through. **A build that completes is not a build that runs** — always run `dist\TradingBotV3\TradingBotV3.exe --selftest` and expect `selftest OK: N/N checks passed (frozen)`, N compared against the current unfrozen count.

- Guards: `tests/test_packaging_spec_drift.py` (every top-level `scripts/` package in `collect_submodules`, every non-`.py` asset under a `datas` rule; deliberate omissions in `PACKAGES_NOT_IN_THE_BUNDLE`) and `launch_gui.py --selftest` (`scripts/selftest.py`, imports every lazily-loaded engine). The two lists must stay disjoint.

- **Triggers:** (1) a new third-party dependency — not covered by the guards; (2) a non-`.py` runtime asset outside the first-party trees plus `config/`; (3) a new top-level package under `scripts/` imported lazily; (4) a dynamic import by string name in an uncollected package (add it to `selftest.LAZY_ENGINE_MODULES` only if a frozen run can reach it); (5) anything touching `__file__` / `ROOT_DIR` / `sys.path` — `ROOT_DIR` is `sys._MEIPASS` when frozen.

### Also moved on 2026-09-13: the last cut

Shortened in the same pass; originals verbatim.

- A SNAPSHOT over 64 MB is stored whole but never `json.loads`-ed; the UNCHANGED watermark is answered from a chunked hash (BD-73).

- Veto vocabulary is versioned and codes are never reused; cohort identity on write is `(vocab_version, reason_code)`, rows are never rewritten, pooling happens only in `_rebuild_pooled_performance`. **Never assert a literal `vocab_version` in a test.**

- Feed repetition control is display only and withholds nothing: one live row per symbol+side+day, repeats fold with an ×N badge, privileged output bypasses the fold; the backing list is written BEFORE any repetition decision. **No suppression field exists in this chain.**

- "Holding highs" is measured in ATR (1.0 ATR, never a percentage) and expires after 15 minutes from the later of the alert and the last new extreme, deleted from the review queue only. Uncertainty never deletes. With-trend rows auto-join M5 Focus (`scripts/regime_pause_focus.py`, DESK only).

- **An armed alert expires in TRADING days** (5 for a 5d extreme watch, 10 for a 20d one and for everything else), counted by `market_calendar.trading_days_between`, policy once in `scripts/armed_alert_expiry.py`. Uncertainty never deletes; every expiry appends a row; a price alert is DISARMED, never deleted. Expiry runs at the head of the poll that owns each store.

- Auto-mode matrix (`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`): discovery is identical in every mode; what changes is who is present. DESK adopts staged picks immediately; AWAY stages, never adopts, and does NOT accumulate a review queue (its return surface is the EOD recap); EVENING runs the early slot, strength checks and briefing, then stops; OFF does nothing automatic.

- A human-focus pick is identified by its CATEGORY as well as its name: `human_focus_tracking._pick_key` is `(trade_date, symbol, side, category slot)`, the slot being the base source with any like-origin suffix stripped. Every join over these files uses `pick_source_family`; a walkaway replays ONE position per (date, symbol, side).

- **The tax number is the BROKER's**: `journal_tax_report` sums `raw_executions.net_amount`, recomputes nothing, refuses rather than estimates (open, `SYNTHETIC_OPEN`, amount-less and unbooked-FX positions are EXCLUDED and named), and shows the recomputed figure beside it, never blended.

- The setup tracker is mirrored into a SQLite record store after every JSON save (`scripts/tracker_store.py`, `tracker_storage_shadow` default ON, never able to fail the save) and the JSON is still the truth; no reader loads from SQLite until gate #57, then readers move ONE AT A TIME (decision 0017).

- The overnight runner's `veto_cohort_grading` slot is deterministic and calls no model. **Stage order is decision 0018's: deterministic slots, then the digest, then narration, then model-gated slots**; a later phase appends inside its stage and never reorders across stages (`EXPECTED_SLOT_ORDER` in `tests/test_ai_jobs_runner.py`). Nothing in this chain may reach a detector, score, alert, watchlist, Focus, the review queue or `review_policy.json`.

- `entry_index.json` is the deterministic compact handoff written beside the packs at the end of `run_daily_digest` (temp-and-rename; a failure never fails the digest): four sections never merged, `changes_vs_prior_window` by FLOOR STATUS only with both pack counts, trials UNRANKED, every `pack_path` the file the numbers were READ from (newest superseding sibling). `repo_commit()` resolves HEAD through a `gitdir:` pointer because agents build in worktrees.

- D1 charts carry a volume underlay and an earnings ribbon drawn INSIDE the price view; neither votes on the price range; earnings headroom is reserved for every symbol; the next report is projected and labelled `est`. Payloads and `scripts/chart_levels.py`'s `levels` are built on the ChartDataService worker, never the paint path.

- The group RS/RW tape owns its own clock (`scripts/group_rrs.py` + `ui/services/group_tape_service.py`): ONE batched `yfinance` download per 5-minute tick, no retry inside the tick, zero IB traffic, no `legacy.py` change; a window without `length + 2` completed same-date M5 bars is `None` and draws nothing. The RS Window tab still reads `rrsSnapshotChanged`.

- **Every ticker click on the Trading Desk charts into the centre Visual Alert Review pane** through `chart_symbol`, never `_enqueue_review_alert`. **A board chart holds NO place in the waiting list and is never re-queued or skip-counted** (`_is_manual_chart_look` on `MANUAL_CHART_TAG`); looking at a WAITING name takes it out for good. `show_board_symbol` is the popup door for a board on ANOTHER page. The `set_chart_sink` wiring: DESK_INTERNALS "Every ticker click lands on the centre chart".

- **Evidence stores are never allowed to cost the thing they record** — a failed append loses the event, never the pick, tracker save or trade. **The one exception is a journal WRITE, which fails loudly.**

- Ground rule 10's statistics contract lives once in `scripts/evidence_stats.py`; `outcome_semantics.claim_kind` decides what may be averaged as a trade (59% of the outcome store is annotations).

- The Market Journal is what the trader thought; the Journal is what they traded — two stores, never merged, both through `shared_journal_service()`. An entry is never backdated: `written_after_the_session` is COMPUTED. A capture joins by `entry_id` from outside.
## DR - the recap reads the day from the stores, not the process (2026-09-13, WISHLIST 10F + 5F)

### The defect that decided the shape

`ui/app.py`'s `_feed_away_recap` handed the AWAY Recap `center._alerts` +
`center._d1_alerts`: the Alert Center's own backing lists, capped at
`MAX_FEED_ITEMS` (250) and `MAX_D1_FEED_ITEMS` (100), and scoped to the PROCESS.
The page's own docstring named the limitation rather than papering over it - a
desk restarted mid-session, or left running across midnight, reported what that
process had seen. On a day the desk restarted at 11:00 the recap's honest answer
was "the afternoon", and on a busy day the morning fell off the end of a 250-item
cap. That is the wrong shape for a record of a day: the evidence was on disk the
whole time.

So `scripts/daily_recap_reader.read_session` takes its whole input as PATHS
(`RecapSources`, twelve durable stores), a session, a lookback and a clock. The
durability property is provable the only way it can be: the test re-reads the
same fixture in a SEPARATE INTERPRETER and compares the `repr` of the whole
session. That is why every emitted moment is pinned to its own fixed offset - a
platform-local `tzinfo` object would still be correct, and its repr would still
have matched here, but the fixed offset makes the equality structural rather than
lucky.

### The four numbers, and the four ways to get them wrong

1. **The store already did the side arithmetic.** `intraday_bounce_outcomes.csv`
   names its side column `direction`, not `side`, and its `mfe_pct` /
   `eod_move_pct` are ALREADY side-adjusted. TSLA (short) and NVDA (long) both
   closed 95.00 from a 100.00 entry; that is `+5.00` for the short and `-5.00`
   for the long. A reader that recomputed `(close - entry) / entry` files them in
   the same place, and a reader that adjusted the stored value AGAIN inverts one
   of them.
2. **Credit starts at the decision's own timestamp.** SHW ran 5.00% the trader's
   way across the session, all of it from the 06:35 bar's 323.00 low. The pass
   was taken at 12:30. The number the trader may be shown as "what you gave up"
   is measured from the LAST COMPLETED bar at the decision (12:25, close 340.00)
   forward to the post-decision low of 333.20 - 2.00%. The day's 5.00% is still
   reported, in its own column, as the day's path. They are two questions and
   they never share a cell.
3. **Unmeasurable is `unavailable`, never the day's number and never 0.0.** The
   MU like at 12:30 has no bar series behind it - the stores record WHAT moved on
   the session, not WHEN - so its credited cell is empty with a reason. Handing
   it MU's 8.00% day MFE would be a claim about timing nothing recorded; handing
   it 0.0 would be a claim that nothing happened.
4. **Present-and-empty is not zero.** An `open` outcome row, an immature horizon
   and an unlabelled session each write their columns present and EMPTY. AMD is
   counted, shown and named unmeasured; MSFT's 3-session end is `pending` and
   out of the rows rather than a pick that went nowhere.

### The window is counted on the exchange calendar

Labor Day 2026-09-07 is not a session, so the third prior session of 2026-09-10
is 09-04. A window counted in calendar days says 09-07 and sweeps in whatever was
scanned on the 2nd. The 1/2/3 control is ONE control: it picks the lookback
window AND the horizon reported, because "how far back" and "how far forward" are
one question for a swing pick.

### Every verdict is its own fact

The grain is `human_focus_tracking._pick_key`'s
`(trade_date, symbol, side, category slot)` PLUS the verdict. KBR was vetoed and
thrown back for the day; STT was vetoed and disliked. Those are two statements
about one name and pooling them counts one decision twice. Two clicks on ONE
statement - the MU claimed like at 12:30 and again at 13:05 - link into one row
with `occurrences = 2` and credit from the first. One pass carrying two reason
codes is ONE decision whose cohorts overlap and are never summed. A retraction
removes a swing favorite. `unfavorite` is never graded at all, and a note is
neither an endorsement nor a rejection. WS-5B's report supplies the journal R and
P&L, joined on the CHANNEL (`pick_feedback:not_today`, `annotation:veto`, ...) as
well as the name, so a vetoed-and-thrown-back symbol cannot inherit the other
statement's match; MFE, EOD return and journal P&L stay three columns.

View 4 shows a refusal only where the later path was favorable, with the ADVERSE
movement beside it and the trader's own reason with it: STT's veto is out because
STT's MFE was 0.00, and KBR's is in with `+4.00` and `-1.50` side by side,
because a later rise alone does not prove a timing or risk refusal wrong.

### Tabs, and why not one flat page

The Strength window's one flat page is the precedent for a set of SMALL boards.
Four recap tables each want a screen of rows, and stacked they put the fourth
below the fold at the desk's windowed 1640x980 - measured, not guessed: the
render test asserts no table runs past the bottom of the page at 3456x2160 AND at
1640x980. One tab per view, each with its population sentence and a sort control
offering only the measures the reader declared (no `mean_r`, `win_rate`,
`expectancy`, `score` or `rank` - the same refusal gate #43 puts on the narration
view). The AWAY staged-pick block sits under the tabs, unchanged: AWAY stages and
never adopts, the page only ASKS, and the R2 gate is shown at click time rather
than enforced.

### What is NOT here

No push. The Daily Recap is a page on the desk, so DESK, EVENING and OFF gain no
routine output from it and the two push exceptions are untouched.
`away_recap.build_recap` and `autopilot_today.txt` are untouched and the AWAY
digest panel is still handed the Alert Center's backing list when the recap page
is selected - the phone's text digest is the same digest it was. The reader
opens no tracker JSON (1.1 GB; a recap that opened it would freeze the desk) and
`alert_center_panel.py` was not edited at all: the chart widget has no marker
seam, so the decision time travels in the row's own Time column and its tooltip
rather than one being invented. A row click goes through `show_board_symbol` -
the door for a board on another page - so a recap row is a board look and takes
no place in the waiting list.

**Tests:** `tests/test_ws_dr_daily_recap.py`. One assertion in it is knowingly
red and is NOT weakened:
`test_the_desk_charts_a_recap_row_through_the_board_door_and_requeues_nothing`
replaces `center._enqueue_review_alert` with a recorder and asserts it is never
CALLED, and on a real `MainWindow` that recorder also catches the scanner's own
`Scanning paused.` status row (`symbol=''`, `side='WATCH'`) arriving from the bot
thread on the first `processEvents()` - with no recap involved at all. The real
method drops a symbol-less alert on its first line, so nothing is queued; the
builder-added `test_the_recap_chart_request_adds_nothing_to_the_waiting_list`
measures the waiting list itself and is green.

**Reopen trigger.** The chart gains a marker seam (then the decision time is
passed to it instead of sitting in a tooltip); WS-10A's dated priority-report
copies land (then the recap reads them instead of `unknown`); the trader asks for
a fifth view, a different default sort, or for the recap to say anything about
money it did not read from the journal.

## WL - one watchlist, many owners (2026-09-13, WISHLIST 10G)

The trader's brief: *"the main Watchlist tab belongs on Trading Desk"* - one list with
views, source badges, side and horizon, *"without turning source into priority"*, and the
Journal linking to its Positions view. The desk already knew every one of those names; it
just knew them in five places, and none of the five could see the other four.

**The rows are ONE pure function and every view is a filter over it.**
`scripts/watchlist_views.build_watchlist_rows` takes the four plain lists, the Focus
store, today's swing favorites, the journal's open trades, the WS-5D intent stream, the
M5 board, today's decisions, the armed price alerts and the per-broker last sync, and
returns `WatchRow`s. It opens no store, writes nothing, reaches no network and reads no
clock it was not handed - so the Qt service can run it on a worker and a test can run it
without a desk. Row identity is `(symbol, side)`: "one symbol may appear once per side".

**Presence on a shared list is not authorship.** `FocusPickStore.add` INJECTS its pick
into `longs.txt` / `shorts.txt`, so "the name is on the list" cannot mean "the trader
typed it" - measured on this branch, `focus_service.add_many(["MU"], "long", "m5")` turns
`longs.txt` from `['AAL']` into `['AAL', 'MU']`. Authorship is the WS-5D stream's answer:
the newest `add` for that `(list, symbol)` names its writer, and `machine_inject` is not
the trader. Where the stream never saw the pair - every pick older than WS-5D - the
fallback is whether a LIVE Focus pick explains the injection; a name nothing explains is
the trader's, because the four files predate every writer that labels itself.
`first_seen` is the earliest add the stream can VOUCH for (`trader_edit`,
`trader_paste`, `machine_inject`); an `observed_external` add is when the desk noticed a
file had changed and leaves it blank rather than back-dating a moment nobody measured.

**Three departures from the packet's wording, all forced by the data.** `positions` is a
TUPLE - the trader has four accounts and one name is legitimately held in two, so folding
them would hide an account or invent a sum nobody holds. `horizons` is a frozenset -
`MSFT` can sit on `longs.txt` AND `swinglongs.txt`, and the horizon cannot be part of a
key that is `(symbol, side)`. And there is a SEVENTH source, `alert`: the price-alert
board retired with the Focus Picks page, so an armed alert on a name that is on no list
would otherwise still be polling and have no row anywhere. An option position's
`exposure` is `None` - "not measured" - because an option's average price is per contract
and this module will not invent a multiplier.

**A position is a read-only projection and it arms nothing.** `stale` is "the last
VERIFIED import run for that broker (`import_runs.status = 'OK'`; `RECONCILE` is a repair
pass, not a look at the broker) is older than the previous session's close", and an
unknown sync is stale too: the row is SHOWN, greyed, never removed, because uncertainty
never deletes. A CLOSED position leaves the automatic view only when the sync that says
so is verified-fresh - a stale close is still shown. A hand-added name survives its
position going away; it loses the badge, not the row. Nothing on this tab writes a price
alert because a position exists.

**The split across the thread boundary.** The Qt thread owns the Focus store (its
`reload()` expires the m5 lists and repairs the fade clocks - it is a WRITER), so it
freezes three in-memory list copies into a `FocusSnapshot`, gathers the cheap stores
(four short text files, two small JSONL, one JSON, the mtime-cached day ledgers - under
3 ms measured) and hands the worker a finished payload. `watchlist-tab` publishes those
rows immediately and only then reads the Journal - sqlite over a year of fills, ~30 ms
here on a warm store and a schema check when cold - and publishes again. Waiting for the
second before showing the first would leave the tab blank for all of it. The service's
15-second tick `stat()`s five paths and starts nothing when none of them moved.

**Every verb still asks the owner it already had.** `WatchlistEditorPanel` for a manual
add, paste or removal (new public `add_symbols` / `remove_symbols(reason=)`, so the save,
the one-name-one-side rule and the WS-5D intent row all keep happening in one place);
`FocusService.remove_everywhere` for a Focus pick; a `swing_favorites` RETRACTION row for
a favorite; `FocusPickStore.restore_faded` - never `discard_faded`, which clears the
entry WITHOUT putting the pick back - for a faded one; `PriceAlertService.save_entries`
for arming. A position row has no Remove at all (`can_remove()` answers before the button
is drawn). The one deliberate behaviour change: the retired board's **Remove** DELETED an
entry, and the tab's **Disarm** does not - A2 says a price alert is disarmed, never
deleted, so the levels and the history stay and "Re-arm" is one click.

**The two retired pages, and why the shortcut had to move.** Chart Review and Focus Picks
are no longer `PAGE_SPECS` rows. Both panel CLASSES stay and both objects are still
built: `ChartReviewPanel` is the annotation rail's reference implementation and other
code imports it, and `FocusPicksPanel` still receives BounceBot alerts, the RRS snapshot
and the mover flags on the desk. `Ctrl+L` - Chart Review's lookup key - is rebound at the
new tab's scope, exactly once, and the tab does two things when it is raised: it makes
the setups column visible (a tab in a hidden column cannot be read, and a `QShortcut` in
a hidden widget never fires) and it moves focus INSIDE the panel, because the widget that
holds focus after a tab switch is the tab BAR, which is not a child of the panel the
shortcut is bound to, and `WidgetWithChildrenShortcut` would not match.

**The journal read never CREATES the journal.** `JournalStore()` builds its schema on
construction, and the trader's own "Prepare Journal database" is a backup, a migration
and a rebuild they are ASKED about. A watchlist read that quietly brought the database
into existence would skip all three - and it did, until `read_journal` was made to refuse
while `store_needs_preparation()` is true; the symptom was
`test_qt_journal_panel.py::test_migration_failure_stays_visible...` failing in the full
suite and passing alone, because the panel found a prepared store where the test had
arranged for none. A journal that is not ready answers with an empty snapshot that says
so, and the Positions view is empty rather than wrong.

**Tests:** `tests/test_ws_wl_watchlist_tab.py` - 55 written red by the tester before any
of this existed, plus two added by the builder. One of the tester's is left RED on
purpose: it asserts `longs.txt` is empty after the tab removes `AAL`, and the same
fixture's Focus add has injected `MU` into that file. Writing `[]` would delete the
injection behind the Focus store's back and BounceBot would stop watching a live Focus
pick; the added test pins the measured behaviour instead.

**Reopen trigger.** The trader asks for the old `Watchlists` file-editor tab to be
renamed or folded in (two tabs a letter apart is a real cost), for a Positions view that
shows money rather than quantity, or for the Watchlist to be a nav page of its own rather
than a tab in the setups column.

## D1C-L - M5 left, D1 right (2026-09-14, packet D1C-L)

**The trader's two words, and which one wins.** 2026-08-31: *"at the end of the day I
have a list of my top swing targets... put it at the very bottom of the M5 alerts tab,
the tab is so long and I never use all of it."* 2026-09-14: *"Keep M5 trades and entries
on the left. Put D1 picks and their management on the right. Inspect the existing
swing-favorites strip, which currently sits below the left M5 list, and reconcile it with
this layout without losing its actions."* The second SUPERSEDES the placement in the
first - the same trader, the same strip, a later instruction - and nothing else about the
2026-08-31 entry changes: the two writes, the `vetted` like-origin, the retraction row and
the "took" badge are untouched by the move.

**The shape.** `scripts/ui/panels/trading_desk.py` now builds two columns. `m5_column` is
a ONE-CHILD vertical splitter holding `m5_alert_bar` (with the ST6.4 Working-lately line
still mounted inside the bar, above the list). `d1_column` is a vertical splitter holding
`master_workspace` on top and `swing_favorites_bar` under it, non-collapsible, the setups
taking the stretch and the strip none. Workspace mode's horizontal splitter is
`m5_column`, `alert_center`, `d1_column`; tabs mode's "Master AVWAP" tab holds
`d1_column` and its "M5 alerts" tab holds `m5_column`.

**Why `m5_column` stayed a splitter with one child.** Every mount, rescue and floor in
the file names `m5_column`, and two shipped tests read `m5_column.widget(0)` for the bar.
Keeping the wrapper kept all of those seams honest and kept the settings code at its
simplest: one child means there is no split to save, so `M5_COLUMN_SPLIT_KEY` is no
longer applied, tracked or written by the desk. The constant survives as the NAME of a
value that is deliberately left alone in `local_settings.json` - the trader's old drag is
not deleted, just never replayed. The D1 split has its OWN new key,
`D1_COLUMN_SPLIT_KEY = "qt_d1_column_split_sizes_v1"`, weights `D1_COLUMN_WEIGHTS =
(6, 1)`, through the same `desk_layout.apply_saved_sizes / track_preset / persist_sizes`.
The opening weights are 6:1 and not 4:1 because the reviewer measured the strip at 399 px
for about 150 px of content at the trader's own 3456 x 2160 - the chip area keeps its
floor and no ceiling either way, and the first drag replaces the preset for good.

**The column is what hides, never the workspace inside it.** `set_setups_visible`, the
open-hidden state and F9's reveal all act on `d1_column`, so the strip comes and goes
WITH the setups: the trader opens that column to look at D1, and a strip out of sight
while the column is hidden is the intended behaviour, not a failure.
`_detach_mode_panels` rescues `d1_column` and NOT `master_workspace` - detaching the
workspace would pull it out of the column and leave the column holding the strip alone -
and `_apply_column_floors` puts the 420 px floor on the column the desk splitter holds.

**A door that pointed at the old widget.** `show_watchlist`'s tabs-mode branch called
`setCurrentWidget(self.master_workspace)`; the tab now holds the column, so that call
would find no tab, raise nothing, and the Journal's "Positions on the Watchlist" button
would do half its job in silence. It asks for `d1_column`.

**A mode switch is not the trader seeing the strip.** `_detach_mode_panels` leaves the
column parentless for a moment. Calling `setVisible(True)` on it THERE shows it as a
top-level WINDOW, and every child gets a real `showEvent` - which fires
`SwingFavoritesBar.firstShown`, whose one job is to re-derive the day's list from the
store. On a desk that had chips on screen, a workspace<->tabs round trip threw them away.
The visibility calls now come AFTER `addTab`, when the column has a (hidden) parent. This
is the same class of defect as replaying a saved split onto a layout it was not dragged
for: a layout change that quietly costs the trader something they typed.

**Tests:** `tests/test_d1c_desk_sides.py` (seven written red by the tester, two added by
the builder), plus the re-pointed place-pinning tests in
`tests/test_qt_swing_favorites.py::TestWhereItLives`, one line in
`tests/test_st6_service_and_surfaces.py` (the left column holds one pane now) and one in
`tests/test_qt_m5_alert_bar.py` (the desk splitter's third child is `d1_column`).

**Reopen trigger.** The trader asks for the strip back under the M5 list, wants it
visible while the setups column is hidden, or asks for the D1 column to open by default.

## D1C - a claimed D1 like is a pick, and the chart is done (2026-09-14, packet D1C-A)

**The trader's words (2026-09-14).** *"The left side of the Trading Desk is for M5
trades. The right side is for D1 trades. When I like and claim a D1 setup, it becomes a
ranked pick I can follow in Master AVWAP Setups. I should not have to keep reviewing the
same D1 chart. ... This request intentionally changes the old 'claimed likes place
nothing' rule for D1 claims. Update that contract narrowly."*

**What it changed.** Until this packet a claimed like wrote one annotation row and moved
the chart on. The name was judged and then nowhere: the row sat in
`trader_annotations.jsonl`, which no trader-facing surface reads, and the same D1 flag
fired again an hour later and put the same chart back up. The claim is now also a PICK -
one row in the Master AVWAP setups table - and the chart it was made on is finished with
while the claim is active.

**The store** (`scripts/claimed_picks.py`, `project_paths.CLAIMED_PICKS_FILE`). Append-
only JSONL beside `swing_favorites.jsonl`, whose shape it copies deliberately: one row per
ACTION (`claim`, `drop`, `expire`), never rewritten, replayed in file order with the last
action per key winning; both clocks on every row (`claim_at` tz-aware machine-local,
`claim_at_utc`, and `session_date` market-local), and `path=` on every function so a test
never resolves the live home folder. Identity is `(symbol, side, claimed_setup_id)` - the
same symbol claimed LONG and SHORT, or under two setups, is two picks, because they are
two theses and they grade apart. The queue gate asks a coarser question and gets
`active_keys`, which is `(symbol, side)` only.

A second `claim` of an active key **appends nothing** and returns the existing row marked
`duplicate`. Repeated clicks are one pick, and the caller still treats it as a success:
the pick exists, so the chart is still done with.

**The expiry and removal rules.** A claim is active until the trader drops it (`Drop my
claim` on the row) or `focus_picks.FADE_TRADING_DAYS` (10) TRADING days pass, counted by
`market_calendar.trading_days_between`. Both are referenced BY NAME - the constant and the
clock are the quiet-Focus-pick fade's, so a re-tuned fade moves both together. The two are
NOT due on the same session, and that is the packet's own rule rather than a defect: a
claim fades on `sessions > FADE_TRADING_DAYS` ("more than ten trading days"), which comes
due on the ELEVENTH session, one session later than the Focus fade's `>=` on the tenth. A
claim is the trader's own thesis and gets the benefit of the last day.
`sweep_expired` appends one `expire` row per faded claim, is idempotent, and runs from the
panel's day roll only: the fade can only change when the session does. **A calendar that
cannot answer expires nothing** - `trading_days_between` raises outside its validated
range, every caller of it deletes something when it answers, and uncertainty never deletes
(plan.md sec 5). A veto, a dislike, a day-trade pass or a "Not today" never retracts a
claim (P5): only the verb that made it, or the fade, can unmake it.

**The horizon is resolved explicitly, once** (`claimed_picks.claim_horizon`). The risk the
trader named was real and measurable: `CaptureRail._timeframe` is `"D1"` from construction
and `AlertChartReview.set_alert` never re-pointed it, so every chart in the review queue
told the rail it was a D1 chart - which also meant an M5 chart's like was filed without the
M5 sidecar bars it was made on. So the answer comes from the ALERT and the CLAIM, and there
is no rail in the resolver's signature for a stale value to arrive through:

1. `alert.is_d1` -> `"d1"`;
2. the panel's own `_is_m5_review_alert` answer, handed to the widget as a flag with the
   alert (the widget never imports the panel) -> `"m5"`;
3. otherwise the claimed setup's registry group. **The registry has no day-trade group** -
   `setup_docs.all_setup_docs_by_group()` is Main swing / Earnings cycle / Study /
   Playbook research, all four of them daily theses - so a named family answers `"d1"` and
   `none_of_these`, an unknown id and an empty id answer `""`. The `"m5"` answer comes only
   from the flag, never from a group name, and this packet invented no group.

`""` places nothing and says so (`claimed; horizon unknown - not placed in Setups`).

**The stale-timeframe read is a SECOND bug at the same seam, and it is fixed separately.**
The rail's `timeframe` is not the horizon - it is what the annotation row RECORDS and what
decides whether the M5 bars ride along as a sidecar - so `set_alert` now passes
`bounce.capture_timeframe(alert.timeframe)` into `set_context`, NORMALISED and never blank.
Both halves of that matter and the first review round found both: `set_context` is
`if timeframe:`, so handing it a typed symbol's empty string left the rail on the PREVIOUS
chart's answer and filed a daily look as `M5` with an M5 sidecar behind it; and a live
`BounceAlert.from_callback` alert says `"5m"`, which upper-cases to `"5M"` and misses
`_record_like`'s `== "M5"` compare - so the one path the attachment exists for was the one
losing its bars. `capture_timeframe` lives beside `BounceAlert` because the spelling is the
alert's own, is pure and TOTAL (`"5m"`/`"M5"`/`"5"` -> `M5`, everything else -> `D1`), and
answers `D1` for a 15m or 1h alert because the review pane has no chart of its own for
those - which is exactly what the rail said before it existed. `set_context` keeps
`if timeframe:`, so every other caller is unchanged.

**Save, confirm, then retire.** `AlertChartReview._route_claimed_like` writes through an
INJECTED `claim_writer` (the panel binds it to its own store) and emits
`claimPlaced(alert, claim_row)` only with a row in hand. A failed write emits
`likeRecorded` instead - the like stands, the chart stays, `_review_queue` is untouched -
and the rail reads `NOT PLACED - claimed_picks.jsonl could not be written; chart kept`.
That status needed a seam of its own: `_record_like` emits `captured` and `commit_like`
then writes its own "LIKE SYM - setup" line, so a host's message was overwritten a
microsecond later and the trader never saw it. `CaptureRail.set_capture_status` /
`take_capture_status_override` let a listener's word outrank the verb's for that one
commit - the verb knows the row was written, the listener knows what became of it, and the
second fact is the one to act on.

`AlertCenterPanel._place_claimed_d1` records `like_advance` through the UNCHANGED
`_record_like_advance` (the name is historical and `review_learning.TAKE_ACTIONS` keys on
it), states `claimed <setup> - placed in Setups`, retires through
`_retire_claimed_review`, and emits `claimsChanged`. **`_retire_claimed_review` is a
separate METHOD, not a flag on `_retire_review_alert`.** That body is the PARKING verb: it
writes `remove_today`, adds the symbol to `_parked_symbols`, drops an auto-adopted Focus
pick and runs three early-return branches. A claim is none of those things, and a flag
threaded through that ladder would be one edit away from parking a name the trader had
just said yes to.

**The repeat-review gate** sits in `_enqueue_review_alert` after the parked check and
BEFORE `_is_m5_review_alert` is consulted, so every M5 alert still reaches the M5 bar
exactly as it did. It fires only for `alert.is_d1` scan alerts that are not chart-watch
hits, and keys on `(symbol, side)`: a claimed LONG says nothing about a SHORT thesis, and
an armed chart-watch is a condition the trader is waiting on. `_active_claim_keys` is an
mtime+size+day keyed cache - one small read when the file changed, never one per alert,
because an alert burst is exactly where a per-alert read would be paid for. The day is part
of the key because the fade is a session clock. Everything upstream of that line is
untouched: the backing list, the feed, History, the D1 badge, the evidence streams, the
AWAY recap and the phone push are all written before it. **This is a display decision that
withholds nothing** - the repetition-control precedent - and the skip is COUNTED
(`_claimed_d1_skipped`) and stated on the review pane the way the movers-only hidden count
is. Nothing is written to `review_policy.json`, which still has no suppression field.

**The row** (`ui/services/claimed_setup_rows.merge_claims`, pure). A claim that matches a
scan row - same symbol, side and setup FAMILY - is a LABEL: the row gains the bucket key
`claimed_like` and the badge `My liked trade` and keeps its own score, bucket and rank,
because the scan measured it and the like did not. A claim the scan does not carry becomes
a NEW row with `score=None`. **A like invents no score and grants no status:**
`priority_score` at claim time is not a scan score, and the Score cell of a claimed-only
row is blank. `known_at_claim` is carried BOTH nested (the honest record of what the desk
had) and flattened onto `raw` (where `setup_points.score_row` looks) - two readers, two
shapes, one set of numbers. What is deliberately NOT flattened is a `setup_family` derived
from the claimed id: the point system reads that field as a MEASUREMENT (a family named
`..._bounce` scores `BOUNCE_NAMED` on its name alone), so a claimed name the scan never
carried leaves it absent and scores +10 - clean path and nothing else - with the notes
naming every unmeasured part. `none_of_these` maps to no family and therefore matches no
scan row; it becomes its own row rather than silently labelling one it was never about.

**One row, all its labels.** `SetupRow.bucket_keys` is `{bucket} | raw["bucket_keys"]`, and
`data_feed._merge_classification_badges` now records the folded row's BUCKET as well as its
label. Until this packet the fold merged display labels only, so nothing recorded that an
HC row was also a FAV and no filter could ask - which is why "FAV + Liked" could not have
been built on the old row model at all.

**Five chips, any combination** (`FAV` / `HC` / `Near` / `Liked` / `All`), replacing three
exclusive selections that could express three views and no others. A row passes when
`row.bucket_keys & selected` is non-empty; `All` means no bucket filter; no chip checked
reads as `All`. Persisted under `qt_setups_bucket_chips` (a sorted list) with a ONE-TIME
migration of `qt_setups_bucket_filter` - each old value gains `claimed_like`, because a
trader looking at their favourites wants the ones they claimed themselves in the same view.
An empty stored list is a real answer, so it is the ABSENCE of the new key that triggers
the migration.

**Ranking.** `setup_points.RANKED_BUCKETS` gains `claimed_like`, so with the Points switch
ON a claimed pick is ordered by the SAME four inputs as FAV/HC/Near - and a ranked row with
no total still sorts after every ranked row that has one, so a claim never jumps a measured
pick by being unmeasured. With the switch OFF, claimed-only rows follow the scan's rows in
claim-time order, newest first. `autopilot_core.order_swing_picks` (the AWAY digest) is NOT
changed: it ranks the SCAN's picks, and a claim is not a scan pick.

**Drop my claim** is registered on the setups table UNCONDITIONALLY, outside the
`focus_service is not None` block beside it (lead ruling): dropping a claim touches no
Focus store, so it must not need one to exist. The star and the X are untouched.

**Tests.** `tests/test_d1c_claimed_picks_store.py`, `_route.py`, `_queue.py`, `_panel.py` -
83 collected. 73 were written RED by the tester before any of this existed; the builder
added ten. Four in the first pass: the skip-count display the packet left to it, the
"exactly one verdict and no rejection" invariant, the proof that the pre-packet advance
route records the next chart's impression too, and a drop on a LABELLED scan row. Six in
the reviewer's fix round: the timeframe normaliser itself, a typed symbol charted after an
M5 alert, a real `from_callback` `"5m"` alert, a D1 flag, a claim `json.dumps` refuses, and
the menu that offers "Drop my claim" only where it can act.

**Reopen trigger.** The trader asks for a claimed pick to reach Focus or a watchlist
automatically, for the fade to be a different clock from the Focus fade, for the queue gate
to key on the setup as well as the side, or for a claimed row to carry a score of its own.

## D1C-B - the claimed picks are graded where likes already were (2026-09-14, packet D1C-B)

Trader, 2026-09-14: *"Reuse the existing outcome and evidence services. Measure manually
claimed opportunities from the claim time forward, retaining the claimed setup and the
measurements known then. Let me compare FAV, HC and My liked trades, and compare the setup
types I claimed. Show sample sizes, pending and unmeasured results, and the existing
uncertainty measures. Handle overlapping buckets explicitly. Repeated clicks must not create
extra independent trades. Keep opportunity results separate from actual journal trade
results. Use these results to show what has worked best over time. Do not automatically
change ranking weights, detector rules or promotion status."*

**No second pipeline was built.** Both sides of this readout were already graded forward
before `scripts/claimed_pick_evidence.py` existed, and neither is regraded by it. The LIKE
side is `ui/annotations/like_cohort.py` delegating to `human_focus_tracking`
(`HORIZONS = (1, 3, 5, 10)`, `entry_close` the claim day's close), written nightly by the
deterministic slot `ai_jobs.cohorts.run_like_cohort_grading`. The TRACKER side is
`master_avwap_tier_outcomes.csv` through the ONE reader
`swing_evidence.read_eligible_rows(..., POLICY_SCANROW_V1)`. The new module reads four files
and writes none.

**TWO CLOCKS, AND THE SAME NUMBER MEANS TWO DIFFERENT THINGS.** Both sides are measured at
`evidence_stats.SWING_HORIZON_SESSIONS` (5), but the like cohort counts five EXCHANGE
SESSIONS from the claim day's close and the tracker counts five SCAN ROWS - the symbol's own
fifth later scan row, which on a name the scan misses for a week is a longer span of calendar
than five sessions. So the two n's sit side by side, each labelled with its own clock in the
cell, and **no number anywhere is their sum**. The tab test that proves it asserts that
`n_liked + n_fav` appears in no rendered cell.

**An overlap is NAMED, never merged, and it is the SCAN'S FACT.** `also_fav` / `also_near` /
`also_hc` answer one question: *did the scan ALSO carry this symbol and side in that bucket on
the claim's own session?* **"That day" means ANY tracker row for the name and side with
`scan_date == session_date`, at any horizon and whatever its eligibility** - the index is
built by `tracker_sightings` from the raw tier rows, not from `EligibleRead.rows`. A liked
pick that matches is counted once in My liked trades, once in FAV, and once in `also_fav`; the
report prints `of which N also FAV that day, M also Near that day, K also HC that day` and
states the basis beside it. A union of two populations on two clocks is not a sample, so there
is no union.

The first cut read the ELIGIBLE horizon-5 rows and the reviewer reproduced the consequence on
copies of the live stores (2026-09-14): FAV 18 against a true 22, Near 17 against a true 31,
and **all 14 misses in the newest sessions**. A claim made this week has no fifth later scan
row yet, so the tier file carries it only at horizon 1 - the rows the trader is actually
looking at were the ones that could never match. Eligibility still governs the FAV / HC / Near
POPULATIONS, which are a different question ("what did this bucket's graded record do?") and
are still answered by `read_eligible_rows` alone. Two tests pin it: a liked row whose only
tier row is horizon 1 IS counted, and the count equals the plain join over the raw tier rows
in both windows.

**HC is UNMEASURED in words and never 0%.** `priority_bucket` in the live tracker, read
2026-09-14, is {`favorite_setup` 8,258, `near_favorite_zone` 16,299, blank 1,809} over 26,366
rows and carries NO `high_conviction` row: HC is an overlay computed at feed-write time
(`legacy._priority_is_high_conviction`) and never stamped on an outcome row. The cell reads
`unmeasured: the tracker records favorite_setup / near_favorite_zone only (0 HC rows)`. The
same reader grades HC the day the tracker ever stamps it - there is a fixture that proves it.

**Three lead rulings, 2026-09-14, on what the tracker's cells may say.**

1. `POLICY_SCANROW_V1.immature_value` is empty - a v1 outcome row cannot exist until the
   symbol's own later scan row does - so `read_eligible_rows` never yields a pending row for
   it. The tracker cells print `0 (mature by construction)` rather than a bare zero, print
   `-` for `unmeasured` rather than 0, and the READ-LEVEL exclusions (`EligibleRead.excluded`,
   e.g. `stale_horizon 1, wrong_horizon 1`) are printed ONCE as `tracker read excluded: ...`
   instead of being repeated per bucket. A dash says "not asked here"; a zero would say
   "asked, and the answer was none". My liked trades keeps its own real pending and
   unmeasured counts, because this week's claims genuinely have not reached their fifth
   session.
2. **`all` is an EXPLICIT wide window, and it is the ONLY one.** `read_eligible_rows(end=)`
   moves only the RIGHT edge of the lately window, so the `all` group passes
   `window=("0001-01-01", as_of)`. `lately` passes `end=as_of` and nothing else, so
   `POLICY_SCANROW_V1.window_sessions` owns the LENGTH and this module never restates how long
   "lately" is; both sides of the report then take their dates from `EligibleRead.window`, so
   the liked rows and the tracker rows can never be filtered over two different spans. The
   tester found the `end=` trap; the test pins it by showing that `end=` alone grades four FAV
   rows where the wide window grades five.
3. The phrase `of which N also FAV that day` lives in `render_text`, with a test.

**Repeated clicks are one trade, twice over.** `like_cohort.like_pick_rows` already keeps the
first claim of the day per `(trade_date, symbol, side)`; the reader de-duplicates again on
`(session_date, symbol, side)` and counts what it dropped in `dropped_duplicates`. Three
`claim` rows for one key in `claimed_picks.jsonl` are one thesis: `_claim_index` keeps the
first. A quick like (P9, Alt+L) names no setup, so it is excluded and counted once in the
footnote `quick likes excluded: N` - never silently dropped. A `like_unclaimed` pick is a
real cohort and a real answer, but it is not one of MY CLAIMED trades and is not read here.

**Ground rule 10: no new statistic.** Every number is a count or
`swing_headline.wilson_lower_bound` (z 1.96). Win rate leads, the sort is the BOUND, and the
leader line names a setup only at `evidence_stats.MIN_REPORTABLE_N` (30) - otherwise it says
`no setup has n >= 30 yet (best n was K)`.

**The journal is a different surface.** Opportunity results and actual trade results are
never merged: this module imports nothing from `journal_store`, `preference_trade_outcomes`
or `journal_analytics`, and there are tests that make those three unimportable and still
build both the reader and the tab. What the trader actually traded lives in the Journal and
in the said-vs-did preference report.

**Where it is read.** ONE renderer, `render_text`, printed by
`python -m claimed_pick_evidence [--window lately|all] [--as-of YYYY-MM-DD]`, and a
`My claims` tab on Research > Setup Tracker built through `_make_explained_tab` - two tables
(`claim_population_table` over `CLAIM_POPULATION_COLUMNS`, `claim_setup_table` over
`CLAIM_SETUP_COLUMNS`), the leader line as the status sentence and the footnotes below.
`_make_explained_tab` gained an `extra` slot for the second block; every existing caller is
unchanged. The four stores are read inside `_read_tracker_exports`, on the panel's existing
`ReadWorker`, beside the other fourteen exports - **the CSV/JSONL loads never run on the Qt
thread**, and a failed read degrades to a sentence rather than blanking the page. TWO memos,
both on the same `(mtime_ns, size)` signature: the RENDER memo (`_table_render_plan`) skips
the model reset and column fit for both tables when nothing behind them changed, and the PARSE
memo is the page's own `_load_csv_rows_cached`, which `load_inputs(read_csv=)` is handed so
the 11 MB tier export is parsed once per file version rather than once per refresh - a
measured 0.33 s of worker time per refresh on a page that refreshes on a spinbox step and on
every tab visit. The JSONL claim store is small and keeps the reader's own read.

**Nothing moves.** No ranking weight, `review_policy.json`, detector, alert, watchlist, Focus
entry, promotion status, tracker file or like-cohort file is written, and the like cohort's
nightly slot is untouched. Nothing under `ai_jobs/` changed.

**Tests.** `tests/test_d1c_claim_grading_reader.py` (24), `_tab.py` (11), `_cli.py` (6) -
41 written red by the tester before any of this existed - plus `_render.py` (6) for the three
lead rulings and `_overlap.py` (10, nine of them red first) for the reviewer's blocker and the
seven advisories it travelled with.

**Reopen trigger.** The tracker starts stamping `high_conviction` on an outcome row; the
trader asks for the two clocks to be pooled, for a claimed setup's record to feed a ranking
weight or a promotion, or for this readout to reach the AI evidence package (nothing in
`ai_jobs` reads it today).

## DTR - the day-trade lists are wiped after the close (2026-09-15, trader-directed)

The trader's words, 2026-09-15, after asking which lists are the intraday ones: *"and the longs
and shorts.txt are wiped at the end of each day?"* - they were not - then *"i want all names
wiped at the end of the day. its a daytrade watchlist not a permanent one."*

**What was true before.** `longs.txt` / `shorts.txt` carried over indefinitely. Only the
machine's own slice moved: the morning open scan replaced what Auto Pilot wrote last time
(`merge_autopilot_watchlist` on `autopilot_written`), the 30-minute auto-populate rotated its
owned slice (`rotate_auto_watchlists` on the day-scoped membership file), and BounceBot's
triple-VWAP rule cut any name with four bad closes (`check_removal_conditions`, trader-typed
names included). A name the trader typed and the tape never invalidated stayed for weeks.

**The rule** (`scripts/daytrade_watchlist_reset.py`). Stateless: the file's modification time IS
the record. A list that holds names and was last written at or before the close of the LAST
COMPLETED exchange session (`market_calendar.last_completed_session`, 16:00 ET) is due; a list
written after that close was written for the NEXT session and stays. Consequences that were
checked, not assumed (`tests/test_daytrade_watchlist_reset.py`): a name typed Monday evening
survives Tuesday morning and goes after Tuesday's close; a desk started on Saturday owes
Friday's wipe; Labor Day is not a session; a write exactly at the close is still that
session's; the desk's naive `datetime.now()` is read as local time; the mtime is compared as an
aware UTC instant against the aware ET close (`astimezone`, never a stripped offset).

**The order of writes.** The file first, through `autopilot_core.write_watchlist_file` (atomic
temp-and-rename, designated-writer gated), then one WS-5D `remove` row per name with the new
source `session_reset` - a machine writer labelled as one, so `watchlist_views` drops its
authorship and `observe_list` on the next Watchlist-tab load reconciles to the empty list
instead of inventing `observed_external` removals. A refused write records nothing (a `remove`
for a name still on the file would describe a wipe that did not happen); a failed append costs
the rows, never the wipe, and the log line says `0 of N intent rows recorded`. The emptied
file's mtime is after the close, so the same session is never wiped twice and an empty list has
nothing to do.

**Who runs it.** `AutopilotService._maybe_reset_daytrade_watchlists` on every 30-second tick,
right after `_roll_day_state` and BEFORE the weekend short-circuit and the open scan, in every
Auto mode - it is a day roll like the autolongs/autoshorts clear, not a scan starter, so quiet
hours do not gate it. After a wipe it forgets `autopilot_written` (there is nothing of Auto
Pilot's own left for the morning merge to replace, and a stale record would let that merge drop
a name the trader types before the open) and logs one line per wipe. A desk open at 13:00
Pacific wipes within 30 seconds of the close; a desk closed at the close wipes on its first tick
back. The CLI `python -m daytrade_watchlist_reset` is the trader's own door and a dry run
unless `--apply`. The `local_settings` switch `daytrade_watchlists_reset` (default ON) turns it
off without a code change.

**What is deliberately untouched.** The swing lists (`swinglongs.txt` / `shortswings.txt`) are
permanent by design. The auto lists have their own day-roll clear. `FocusPickStore` and its
injection membership: an M5 Focus pick is scanned through the fast lane whether or not it is on
`longs.txt`, and its later un-injection finds the name already gone and records nothing
(`_uninject_from_shared` checks presence first). BounceBot re-reads both files every cycle
(`self.longs = read_tickers(LONGS_FILENAME)`), so no detector file changed. Nothing here reaches
a detector, score, alert, tier, the review queue or `review_policy.json`.

**The invariant.** plan.md sec 5 said "user-entered watchlist names are never automatically
removed"; that rule was written against a MACHINE JUDGEMENT about one name (the auto-populate
rotation, the cross-machine writer race). Decision 0020 amends it narrowly: the day-trade lists
are emptied WHOLE at a session boundary, every name alike; the swing lists keep the rule as it
was; no writer may still remove one user-entered name by its own judgement.

**Open for the trader.** Whether the M5 Focus picks should reset with the lists (today they fade
on their own ten-session clock, `focus_picks.FADE_TRADING_DAYS`), and whether the 13:00 Pacific
close is the right moment or the wipe should wait for the evening. Gate #123.

## PCT-3 - compression is measured before it is tuned (2026-09-15, packet PCT-3)

The trader's words, 2026-09-15: *"we need a way to measure for compression as the most COMMON
veto I have is a compression veto. we need to fine tune this so we stop getting so many
compressed picks. this will also help us find compression breakouts."* Asked what to build
first, they answered **"Measure + chip first"** - one measure, a chip, a check against the
vetoes, and the penalty decision from that evidence. No hiding.

**The thing that was already true, and invisible.** `legacy.summarize_anchor_compression` has
always computed FOUR numbers over the bars from the current AVWAPE anchor to the last trade date
- `compression_score` (0-3: how many of the three ratios are tight) plus `stdev / range /
close_range` as multiples of ATR-20 - and always thrown them away inside the function. Only
`compression_flag`, `compression_penalty` and `compression_note` reached the priority row, and
the penalty (10 to 22, and more when the structure penalty lands too) was silently subtracted
from the score. So the desk has been demoting compressed setups since before the trader ever
typed the word, and the trader has never seen one of those numbers. That is the whole diagnosis:
the measure and the eye disagree, and nobody could see by how much.

**There are TWO row builders, and only one of them is the desk's.** This is the defect the first
review of PCT-3 caught, and it is the thing to know before touching anything in this area:
`legacy._evaluate_priority_snapshot_for_date` and `runner._run_master_impl`'s per-symbol loop are
near-duplicates of each other, and **the live desk scan runs the one in `runner.py`** - that loop
is what writes `master_avwap_ai_state.json` and the priority rows. `legacy`'s copy is reached only
by `build_completed_bar_priority_rows` (whose ai_state is discarded) and by the tracker catch-up
backfill. PCT-3's first build applied the copy-through in `legacy.py` alone, and the live file
proved it: 1003 symbols, 35 flagged, **0** with a `compression_score`. Anything additive that has
to reach the desk goes in BOTH, and `tests/test_pct3_runner_publishes_compression.py` drives
`runner._run_master_impl` for real (one synthetic symbol, every outside door closed) and reads the
ai_state file off disk, so a future build cannot quietly lose one seam again.

**Item 1 - the copy-through.** `compression_copy_through` publishes the four numbers, with
`compression_rule_version = "anchor_compression_v1"`, on the priority row, the `ai_state` symbol
entry, the feature row and `build_tracker_setup_record` (flat, and as a `compression_summary`
mapping) - at both row builders. The feature row's seven columns are also added to `runner`'s
`feature_columns` allowlist, because `df_features` is built with `columns=feature_columns` and a
key that list does not name is silently dropped: the `assigned_tier` defect, again. It is a COPY:
it reads no bar, decides nothing and touches no score. The penalty on the row stays the
EFFECTIVE one (`_effective_compression_penalty`'s breakout relief applied); the ratios and the
score are the measure's own and are never relieved, because a report that showed a relieved
ratio would be measuring the relief. The tester's four synthetic frames still score
-70.0 / -12.0 / -10.0 / 33.0, and `tests/test_master_avwap_setups.py`'s exact-score assertions
pass unchanged.

**Items 2 and 3 - the chip.** `scripts/compression_chip.py` is the pure reader
(`read_row / read_flag / tooltip_text`), and `SetupTableDelegate` paints an amber `caution`
`compressed` pill AFTER the bucket chip and after WS-WS's `wrong side` chip, with one added
tooltip line naming the score out of 3, the three ratios and the penalty. Display only: nothing
is hidden, nothing is re-ordered, no score moves, and an unflagged row is pixel-identical to a
row that has never heard of compression. The defect class the reader exists for is
`bool("False") is True` - a flag that has been through a CSV, a JSON round-trip or a report line
spells itself a dozen ways, and every one of them reads the same here; an unreadable spelling is
NOT compressed rather than a guess.

**A row that carries no measure reads as NOT MEASURED.** `compression_copy_through_from_row` is
the reader for a stored row, and it answers `None` for the score and the three ratios and omits
`compression_rule_version` entirely when the row carries none of them. A `compression_score` of 0
stamped `anchor_compression_v1` says "this rule looked and found nothing tight" - a measurement
that never happened, and precisely the number a calibration report would average. The same rule
governs the break verdict: `_compression_break_copy_through` stamps `compression_break_recent` and
`compression_break_rule_version` **together or not at all**, because a version beside no verdict
claims a rule ran.

**Where the numbers join the row, on which thread, and where they may not.** The setups table is
built from `master_avwap_priority_setups.txt`, whose ranked lines carry symbol, side, score,
family and bucket and nothing about compression; `master_avwap_ai_state.json` carries the whole
reading per symbol. That file is 36 MB and one `json.load` of it is **281-292 ms**, and
`master_avwap_panel.refresh_from_reports` runs on the Qt thread on the trader's own click AND on
every report-watcher signal - so PCT-3's first build put a third of a second of frozen table
behind every file change. The seam is therefore split in two:

- `ai_state_levels.warm_cache()` does the parse, and only `_AiStateCompressionWorker` calls it.
  `refresh_from_reports` starts that worker behind an `(mtime_ns, size)` signature, so a burst of
  watcher signals costs ONE parse; it returns `True` only when the cache actually MOVED, which is
  the panel's cue to run exactly one more coalesced refresh.
- `ai_state_levels.cached_symbol_compression()` is the Qt thread's door and never opens or stats
  anything. `data_feed.merge_compression_from_ai_state` defaults to `allow_read=False` and reads
  that; `allow_read=True` is for a worker, a CLI or a test, never the Qt thread.

A cold cache is a row with no chip, never a stall. The join is by SYMBOL, because the anchored box
is a property of the symbol's current anchor rather than of the side being traded, and it FILLS: a
row that arrived from the focus feed with its own fresher reading keeps it.
`tests/test_pct3_compression_merge.py` runs the panel's own `refresh_from_reports` with `open`
spied and asserts the ai_state file is never opened on that thread, spies `open()` across a whole
paint pass, and proves that a compression feed which RAISES cannot reach the paint pass at all.

**Item 3's CLI - the report that answers the question.** `scripts/compression_calibration.py`,
run as `cd scripts && python -m compression_calibration --since 2026-08-20 --live`. It joins
every coded compression veto in `trader_annotations.jsonl` to the tracker's population for the
sessions those vetoes fall on, and prints, per measure, `n = vetoed / rest`, the two medians and
a rank-sum AUC (the probability a randomly chosen vetoed row reads HIGHER than a rest row, ties
counted as half). Seven measures: the scan's three anchor ratios, `range10_atr14` and
`range20_atr14` (`d1_environment`'s rule, per symbol), Bollinger(20, 2) relative width as a
percentile over 120 sessions, and ATR-14 / ATR-50. Then the hit rate of today's
`compression_flag` on the vetoed set. **A low AUC on every measure is the finding, not a
failure**; the threshold and penalty change is a SEPARATE ask after the trader reads it.

**Every veto is counted and NAMED, and the population is what was ACTIVE.** The first review ran
that CLI against copies of the live stores and found it silently discarding most of its own
evidence: 213 unique compression vetoes over 12 sessions, **86 joined**. Two causes, both worth
remembering because they are the same mistake in two places.

- It joined a veto to a tracker record whose session equalled the veto's, and a record's session
  is `scan_date` - the day the setup was ENTERED. A veto on a name the trader had been carrying
  for a week matched nothing. The population for a session S is now every record **ACTIVE** on it:
  `record_active_window` reads the record's own `entry_trade_date` and `last_replayed_session` (the
  replay advances that stamp every session a setup is still open and stops the moment it closes or
  expires), and S must fall inside. The header prints that definition; the earlier version printed
  the word "shown", which the report never knew.
- A session that has not closed was dropped without a word - the whole 2026-09-15 session, 23
  vetoes, while the header printed "..09-14". Now every veto lands in exactly one of `joined`,
  `untracked` (no record was active - it still gets the four fixed-window measures and still
  counts on the vetoed side, labelled `untracked` in the CSV) or `pending` (the session is not
  complete, so no measure may be taken on it at all), and the line `N vetoes: A joined, B
  untracked, C pending` is printed above the tables with the excluded sessions named and why.

**What the first full run actually said** (2026-09-15, against copies of the live stores, since
2026-08-20): `213 vetoes: 140 joined, 50 untracked, 23 pending`; 11 measured sessions
2026-08-20..09-14 with 2026-09-15 excluded as incomplete; 14,759 population rows, 190 vetoed;
14,709 anchors recomputed. **Today's `compression_flag` hit rate on the vetoed set is 0.18** - it
caught 35 of 190 rows the trader vetoed for compression, and it was set on 2,631 of the 14,569
rows they did not. Every one of the seven measures scores an AUC BELOW 0.5 (0.34 to 0.49): a
vetoed row does read tighter than the rest on all seven, consistently and weakly, and no candidate
separates the two populations. That is the finding the threshold conversation starts from, and it
is a finding, not a failure. 267 s, peak RSS 0.14 GB.

**An anchor measure the record does not carry is RECOMPUTED, never printed as `nan`.** No live
record carries the copy-through yet, so the three anchor measures came out `n = 0 / nan` - three
empty tables in a seven-table report. `recompute_anchor_measures` runs the champion's own
`calc_anchored_vwap_bands` at the record's anchor date and `compute_atr_from_ohlc`'s ATR-20 at the
session, and hands both to `summarize_anchor_compression` - the same function, never a second
copy. A row whose anchor cannot be had says `anchor unknown` and still carries the four
fixed-window measures. Every row's source is a CSV column (`record` / `recomputed` /
`anchor unknown`), so nobody has to guess which number came from where.

Four more properties of that CLI are load-bearing and each one is a rule:

- **The two veto codes are pooled BY NAME, here.** The plan's premise was that v1's
  `support_resistance_cluttered` pools with v3's `compressed` in
  `veto_cohort.canonical_veto_cohort`. Measured on this branch, it does not:
  `veto_v1_support_resistance_cluttered` stays itself, because `veto_reasons_v2.json` introduced
  `compressed` as a NEW definition and its own description says so ("That is a NEW code, not a
  rename"). So the live "189 + 25 = 214" is a sum this report makes for itself. A veto with a
  blank `reason_code` (136 of the live rows) is a veto and is NOT a compression veto: it counts
  in "rest".
- **Read-only, and it says so before it reads.** `project_paths.DATA_DIR` is read at CALL time
  (the `d1_environment_store._cached_daily_bars` idiom). A path carrying `TradingBotData` - or
  the DAS's `Trading Bot Data`, since spaces are stripped before the compare - is refused
  without `--live`, exit 2, nothing written. The message names the folder, names the flag, and
  names every store the run would read (the veto file, the tracker, and the daily-bar cache under
  `%LOCALAPPDATA%`) with the word READ-ONLY beside them, because a refusal that hides one of the
  three stores is a refusal the trader cannot check.
- **Point-in-time** (plan.md sec 5). The cached daily frame is cut at the session date before a
  single measure is taken, so no later bar can reach a range, an ATR or a Bollinger percentile.
  The proof is the same fixture run twice, once with ten wild sessions written AFTER the
  session, with the two CSVs compared column for column.
- **It STREAMS the 1.26 GB tracker.** `json.load` on that file is one of the three causes of the
  10 GB desk on 2026-08-27, and the first build's `read_text` was not enough either: measured at a
  **3.79 GB** tracemalloc peak. `iter_tracker_records` now feeds a bounded sliding window
  (`_JsonWindow`) from a buffered reader and lets `json.JSONDecoder.raw_decode` parse ONE record
  out of that window, discarding it behind; a non-record section is walked past structurally
  rather than decoded. Peak is a few megabytes above the largest single record, pinned by a test
  that reads a multi-record fixture through a reader which RAISES if asked for more than 64 KB at
  once. The SQLite mirror would be cheaper still and is deliberately NOT read: decision 0017
  fences every reader out of it until gate #57.

**Item 4 - `compression_break_v1`.** `legacy.evaluate_compression_break_v1` REUSES
`assess_compression_break_context`, which is already the one place that decides "the previous
completed session's anchored slice reads compressed AND today's completed close leaves that box
in this side's direction past the 0.10-ATR buffer", and adds exactly one clause: today's own bar
range must be at least 1.0 ATR-20. A drift out of a quiet box on a quiet bar is not a break. It
sets three labelled names - `compression_break_recent`, `compression_break_v1_note` and
`compression_break_rule_version`. It changes no score, gates nothing, and leaves
`compression_break_today`, the Phase-6 study row and `enrich_priority_rows_with_phase6_studies`
exactly as they were; the labelled version is what makes the rule re-tunable by name once the
calibration report is read.

**v1's note has its own field, and the tag is a CONFIRMATION.** Two rules that came out of the
first review. `enrich_priority_rows_with_phase6_studies` does `row.update(context)` and OWNS
`compression_break_note` on the finished row - so v1 writing there was writing into a field that
gets replaced, and on a narrow-bar break Phase 6's note is non-empty while v1 refused. v1 keeps
`compression_break_v1_note` for its own answer. And `setup_tagging` adds `COMPRESSION_BREAK` in
**confirmation order**, beside `TRENDLINE_BREAK` rather than ahead of the confirmations: the
visible tag list is capped at `DEFAULT_MAX_SETUP_TAGS` (6), and a label added in 2026 may not cost
a row a tag it has carried for a year. A test pins a crowded row's tags byte-identical with and
without the new flag.

**What is deliberately not here.** No threshold moved. No penalty moved. Nothing hides a
compressed row, mutes it, re-orders it or keeps it out of a list - the chip is a label and the
report is a report. Nothing here reaches a detector, a score, an alert, a watchlist, a Focus
list, the review queue or `review_policy.json`. The trader's own words set that boundary:
measure first, then tune.

## PCT-2 - a trendline break is a tag and an event (2026-09-15, packet PCT-2)

- **The tag stays a scan fact; the event is an explicit arm.** `TRENDLINE_BREAK` already describes
  the scan's recent break candidate. The new `trendline_break` D1 event is an extension kind, so
  Focus auto-interest never constructs it. It is available only when the trader arms the chart.
- **A line must not move after the arm.** `D1EventWatch` saves the scan candidate's explicit stable
  line id, type, endpoint dates/prices, lookback anchor, current projected price, log slope and
  candidate break date, with side and the compact saved D1 report's actual offset-aware parseable
  `generated_at` as knowledge time. The minute poll reads no report and never consults a redraw; it
  validates every frozen fact before confirming. An old or partial watch still loads, but is
  uncertainty and cannot fire; a missing, malformed or timezone-less report time refuses the arm.
- **Only two completed D1 closes can prove it.** The prior close must be on or inside the frozen
  line and the next completed close must be through it in the setup direction. M5 bars, wicks,
  forming daily bars and a prior close already through do not count. The one-shot watch retires on
  the first hit, and the scan report's identity is `(symbol, side, break_date)` rather than a
  rounded moving level. Duplicate persisted trendline watches are also collapsed to one save and one
  emit for that key in a sweep.
- **The champion report remains untouched except for the new row.** The bucket-upgrade saved-report
  path appends `Trendline break` from the scan's already observed candidate. Its static pre-change
  fixture proves the existing D1 rows survive byte-for-byte; the separate partial `Trendline
  breakthrough` Focus event in `master_avwap_shared.py` is not replaced or altered.

## PCT-1 - the Pullback alert (2026-09-15, packet PCT-1)

- **What the trader said (2026-09-15, chat).** *"we then monitor them for a pullback on a M15 or
  M30 basis. on the M15 we use the 150 moving average on the M30 we use the 75. we wait for the
  stock to go BELOW these levels then break back up. we can test entries on the breakup if they
  are accompanied with an LRSI reversal on the same time frame in the past 3 bars or so. we can
  also test on an M30 basis a breakup then waiting for an M30 or M15 LRSI reversal while staying
  above teh relevant SMA for an entry. we can also test for retests of the SMA ... in the same
  vein as H1 retester we should rename it to Pullback alert and include any of these phenomena in
  the alert pattern."* Their four answers the same day: which names it watches - **"Chart arm +
  my picks"**; what an LRSI reversal is - **"Cross up through 80. Ideally it was below 50 2-4 bars
  previously too"**; and **"Yes, all of them"** to editing the ask-first files for this work.
- **One button, one kind, four named triggers.** `WATCH_KINDS` has `pullback` -> "Pullback alert"
  and no longer has `h1_ema_bounce`; the arm bar builds one button per kind, so the retester's
  button is simply gone. A `ChartWatch` now carries `triggers`, `fired` (trigger -> the bar time
  it last fired on) and `declined`. A row stored as `h1_ema_bounce` loads as a `pullback` watch
  whose ONLY trigger is `h1_ema15_bounce`: nothing the trader armed is lost by the rename and
  nothing they did not ask for is added to it. The H1 rule sheet `h1_ema_bounce_v1` is untouched -
  it is one of the four things this watch waits for, and `evaluate_h1_bars` is called exactly as
  it was.
- **The rule sheet is frozen first, then wired.** `pullback_sma_reclaim_v1`
  (`scripts/indicators/pullback_sma_reclaim.py`) is pure: completed session-aligned bar dicts in,
  a frozen result out, `now` a parameter and no clock read inside. `sma_reclaim_lrsi` fires on the
  reclaim bar when the LRSI crossed up through 80 on it or one of the two before it;
  `reclaim_then_lrsi` is the M30 leg and fires on the later cross - on the M30 itself or on the
  M15 companion series the caller hands in - while every completed M30 close holds the 75-SMA;
  `sma_retest` fires on a bar AFTER the reclaim whose low comes within 0.25 ATR-14 of the line (or
  through it), which still closes on the right side, AND whose own timeframe has an LRSI 80 cross
  on that bar or either of the two before it. **The reclaim bar is never its own retest.** Warm-up
  is the SMA's length plus ten (160 M15 / 85 M30 bars) and 24 h of silence is
  stale; both answer `None`, which is NOT MEASURED, never a verdict.
- **80 is a parameter, never a live level.** The oscillator is the champion's
  `efficiency_lrsi.compute_efficiency_lrsi`, whose `CROSS_LEVELS (20, 50)` are the M5 engines'
  and are neither read nor written here; a SHORT reads the NEGATED closes, the idiom
  `m5_signal_engines.latest_lrsi_cross` already uses, so "cross up through 80" means the same
  thing on both sides. `m5_signal_engines` and `bounce_bot_lib` are untouched.
- **"Ideally it was below 50" is a label, not a gate.** The trader said *ideally*, so
  `lrsi_from_below_50` rides every fire and every `watch_fired` row and gates nothing. It is
  counted back from the CROSS bar (lead ruling, 2026-09-15).
- **An episode is the unit, not a bar.** A completed close back under the SMA opens a new episode;
  each trigger fires at most once inside one and again in the next. The panel carries the episode
  state in memory between polls and the watch's `fired` map is the PERSISTED backstop, so a desk
  restart does not re-announce a move the trader was already told about.
- **A Pullback alert is a STANDING arm, except on the H1 leg.** An SMA fire records its bar and
  leaves the watch armed for the next one, up to the ten-trading-day expiry or a disarm; the H1
  leg keeps its WISHLIST 10C one-shot contract, because a completed retest is the pattern
  finishing rather than a moment inside it.
- **The trader's own picks arm themselves.** Inside the same 60-second poll and BEFORE any
  evaluation, every active claimed D1 pick and every swing Focus name gets a `pullback` watch
  carrying `auto: claimed pick` / `auto: swing Focus`. The sweep owns only what it armed - a watch
  with no `auto:` source is the trader's own click and is never retired by it, the same "absence
  of a marker means the trader owns it" rule Focus provenance holds. When the claim or pick is
  gone the watch goes with one `watch_retired_source_gone` row. An unreadable store arms nothing
  and retires nothing: uncertainty never deletes.
- **A hand disarm of an auto-armed watch is REMEMBERED, not obeyed once.** The sweep runs every 60
  seconds, so deleting the row would put it straight back and the trader could never turn one off.
  So `disarm_chart_watch_for` keeps it with `declined=True` - persisted in the same store, hidden
  from the Armed board, never evaluated, never pushed - until the claim or pick that armed it is
  dropped. A watch the trader armed by HAND is still simply deleted.
- **These push in every Auto mode.** They are the trader's own picks and they ride the existing
  armed sender (`_push_armed_watch` -> `notify_armed_watch`), which is the recorded exception
  class beside the Research/Focus price alerts; recorded as an amendment in
  `docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`. One `watch_fired` review row per fire, carrying
  `trigger`, `timeframe`, `rule_version` and `lrsi_from_below_50` - and, since the review, an
  auto-armed watch's `sma_retest` is a row and a feed line without a phone buzz (see the review
  round below).
- **The M15 and M30 history is fetched, and never on the Qt thread.** The desk has no cached M15
  or M30 series at all, so unlike the H1 leg these two have no primary to fall back FROM:
  `IntradayHistoryCache` (`scripts/intraday_history.py`, the WISHLIST 10C H1 cache generalised by
  `interval_minutes`; `h1_history.H1HistoryCache` is now that class fixed at 60) is the only
  source. One yfinance call per completed session-aligned bucket per symbol, on the cache's own
  worker, zero IB traffic, and since the review ONE multi-ticker download per chunk of 50 symbols
  rather than one per name. The rule is **once per completed session bucket, including the one the
  bell cuts short, and never outside the session's own buckets**.
- **The health cell is one state per timeframe, joined with `;`** - `H1 from cache; not measured
  (0 of 160 M15 bars); M30 from yfinance`. Each H1 string is byte-identical to the one WS-10C
  shipped; only the joining is new, and a watch stored before the rename still reads exactly as it
  did. The M15/M30 halves READ the cache and never fetch: the poll is what asks, so a cosmetic
  string never puts the network on the Qt thread's critical path. An armed surface whose job is to
  say "these are the exact conditions I am waiting on" must not report `ok` beside a leg that
  cannot evaluate at all.
- **Three claim names, and a watch is still never a claim.** `pullback_sma_reclaim`,
  `trendline_break` and `compression_break` join `SETUP_DOCS` under the new group "Entry timing
  and breaks" and `EXTRA_CLAIM_IDS`, so the rail offers them; `setup_registry_v1.json` is
  regenerated by `build_setup_registry.py --write`, never by hand. They are graded by name through
  `claimed_pick_evidence._by_setup` and nothing else is needed for the "My claims" tab. WISHLIST
  10C's fence still holds: an armed watch grades nothing, joins no cohort, and reaches no
  detector, score, tier, gate, watchlist, Focus list, review queue or `review_policy.json`.
- **`claimed_picks.record_drop` is unchanged**, and that is a decision. PCT-1 briefly gave
  `claimed_setup_id` a blank default and made a nameless drop end every claim on a `(symbol,
  side)`; the review sent it back and the lead agreed - a wildcard retraction is a wider
  production semantic than any caller needed, and the test that wanted it simply names the setup
  it claimed.

### What the review round changed (2026-09-15, six blockers)

The first build was measured against copies of the live stores - 24 claims and 54/22 swing Focus
names, so **95 watches on the first tick** - and six things it got wrong are worth keeping
written down, because each is a rule the next standing-arm feature inherits.

- **A STANDING arm needs an event key, not an arm key.** `notify_armed_watch` de-duplicated on
  `watch_id` for the life of the process, which was exactly right while an armed watch fired once
  and disarmed. A Pullback alert does not disarm, so only its FIRST fire ever reached the phone.
  It now takes keyword-only `event_key` defaulting to `watch_id` - every one-shot caller is
  byte-identical - and the pullback push passes
  `f"{watch_id}:{trigger}:{timeframe}:{bar_dt.isoformat()}"`. The tester's push double had hidden
  this: **a fake that only counts calls cannot see a refusal inside the real service.**
- **An action joins the scoring sets on what its WRITER does.** The sweep armed through the public
  button, which writes `arm_watch` - and `review_learning.TAKE_ACTIONS` holds `arm_watch`, so the
  first tick recorded 95 trader TAKEs. The sweep now writes its own `auto_arm_watch`; the retire
  keeps `watch_retired_source_gone`; neither is in TAKE_ACTIONS, REJECT_ACTIONS or
  `watch_conversion`. The hand-armed button still writes `arm_watch` and is still a take.
- **Nothing expensive on the Qt thread, and "expensive" includes ninety-five of anything.** The
  first tick cost 1.92 s there and every tick after ~0.7 s. Three changes: the sweep arms in
  memory and then saves ONCE, appends its review rows in ONE batch
  (`review_events.record_review_events`, one kernel lock and one open) and emits ONCE; a timeframe
  is judged only when a new completed bucket has closed for it since that watch was last judged on
  it (`intraday_last_bucket_end`, the one thing that can change any of these answers); and the
  SMA/LRSI/ATR passes run on a worker, returning fires through the queued `pullbackFiresReady`
  signal so the Qt thread only records, pushes and draws. The H1 leg stays on the Qt thread
  because it reads BounceBot's M5 cache, which is not thread-safe - but it is bucket-gated too and
  paced at `PULLBACK_H1_BATCH_LIMIT` (12) watches a tick, oldest-waiting first. Measured after:
  **140 ms for the arming tick, 16 ms for a warm one.**
- **One worker, batched downloads.** 285 single-ticker `yf.download` calls and 286 threads came
  out of that tick. `request` now enqueues and the cache's ONE worker issues one multi-ticker
  download per chunk of 50 - the group RS/RW tape's own shape. 95 symbols is 2 requests.
- **A bucket the bell cuts short is still a completed bucket.** The first build refused every ask
  after the close, which silently dropped the CLOSING bar of every session while the health cell
  still read `from yfinance`. Removed; `h1_history` is behaviourally identical to what shipped.
- **A remembered refusal the trader cannot undo is a trap.** A declined auto watch left the arm
  button reading ARMED beside an empty Armed board and no way back. `armed_watch_kinds` now
  excludes declined rows, and arming from the chart DROPS the declined row and re-arms as a
  HAND-armed watch (its own `source_text`, a fresh `watch_id`, nothing fired) - so the sweep no
  longer owns it. The expiry pass runs over declined rows too: a remembered row is still a
  ten-trading-day arm.
- **Phone volume is a design decision, not a side effect** (lead, 2026-09-15; the trader may
  overrule). The reviewer measured ~85 fires a session at 108 watches, 70 % of them `sma_retest`.
  A watch the DESK armed pushes `sma_reclaim_lrsi` and `reclaim_then_lrsi` and writes an LRSI-
  confirmed `sma_retest` as a feed row and a review row WITHOUT a push; a watch the TRADER armed
  by hand pushes all three. A held retest without that LRSI cross makes no fire, feed row or
  review row. Nothing else is withheld - every fire is on the feed and in the evidence.
- **`ChartWatch.fired` is keyed `trigger@timeframe`.** Keyed on the trigger alone, one watch's M15
  and M30 stamps overwrote each other and a restart re-announced whichever lost.
## SC - the setups table cycles, and a veto for the day hides the row (2026-09-15, trader-directed)

Trader, 2026-09-15: *"when i click on master avwap setups tab and then I click the veto or
like and claim buttons it should cycle it to the next pick. additionally vetoing it for the
day SHOULD remove it from the list (but the stock should still be tracked for setup tracker
purposes)"*.

**What was true before (recon, file:line verified).** A setups-table row click reached the
centre chart through `MasterAvwapPanel._open_symbol_snapshot` -> `chart_symbol`, which built a
`MANUAL_CHART_TAG` alert; `_select_review_alert` gave such a chart NO place in the waiting list
(packet T1: a look is not a shown alert). Both retire paths - `_retire_after_veto` ->
`_ignore_alert_symbol` and `_place_claimed_d1` -> `_retire_claimed_review` - ended in
`_advance_review_queue`, which pops the ORDINARY waiting list and knows nothing of the table.
So a veto on a setups chart showed whatever M5 / D1 alert was queued next, or "waiting". And
WS-SX's ✕ mark was paint only: `SetupFilterProxyModel.filterAcceptsRow` had no decision
predicate, so a vetoed row stayed in the table with a red ✕.

**The cycle.** `chart_symbol` gained a `next_pick` callback - a caller's own "what comes after
this chart", returning True when it charted something. It is stored on the panel
(`_manual_next_pick`), consumed by `_advance_review_queue` BEFORE the waiting list is read (a
veto, a claim and the Next verb all end there), and dropped in `_select_review_alert` the
moment any non-manual chart takes the pane; a lookup-box or board chart passes none, which
clears an older one. The setups panel passes one for every row it charts
(`_chart_row_on_desk`): `_chart_next_pick` finds the row after `(symbol, side)` in the proxy's
VISIBLE order - by identity first, because a refresh may have moved it; when the hide filter
has already removed the vetoed row, the row now at its old index IS the next one - skips the
same symbol (the other side of a name just vetoed) and any symbol rejected today, moves the
table's selection and scrolls to it, and charts it with a fresh callback of its own. When the
table runs out the status row reads `End of the setups list - nothing after X` and the
waiting list takes over. A callback that raises is logged and the queue proceeds. The waiting
list is never touched by the walk.

**The hide.** `pick_feedback.HIDDEN_REJECT_KINDS` = `veto`, `dislike`, `not_today`,
`remove_today` - the swing-side verdicts. A day-trade `pass` and an M5 click-away are
verdicts on ANOTHER population (CLAUDE.md P5: no two verdicts are combined) and hide nothing.
`DayDecisions.rejected_symbols()` reads the same snapshot that paints the ✕; the panel's
`_on_day_decisions_ready` feeds it to `SetupFilterProxyModel.set_filters(rejected_symbols=)`,
so the hide and the mark can never disagree. The strip's `Show vetoed (N)` box
(`qt_setups_show_vetoed`, default OFF) restores the rows in their original order with their red
✕. Per SYMBOL, as the ✕ mark is: vetoing the LONG thesis hides a SHORT row of the same name too.
Presentation only - nothing is deleted, re-ordered or written; `filterAcceptsRow` is read by
the view alone, and the Master AVWAP scan, the Setup Tracker's save pass and every evidence
writer never see it, so the vetoed name is still tracked (the trader's parenthesis).

**Why the setups table hears the chart's verdict at once.** WS-SX's marks refreshed through the
Focus coalescer and the report poll; a chart veto that touched no Focus pick could wait 30 s.
The Alert Center now emits `reviewDecisionRecorded` after a rail veto and a placed claim, and
the desk connects it to `MasterAvwapPanel.refresh_decisions` (the same 200 ms coalescer).

**What WS-SX loses.** Its item 5 said "a decision moves nothing". The trader amended it; the
test that pinned it (`test_a_decision_marks_a_row_and_moves_nothing`) is now
`test_a_veto_hides_its_row_and_show_vetoed_brings_it_back` and gate #99's red-✕ check is read
with `Show vetoed` ticked. Three WS-SX paint tests are ORDER-DEPENDENT (they pass after
`tests/test_qt_desk_ticker_clicks_chart_center.py` loads the theme and fail alone) - found
here, pre-existing, not fixed.

**Lead decisions the trader may overrule.** The table's own ✕ (a coded dislike) hides too - it
is the same "not today" in the trader's hand; the hide is per symbol; a claimed row stays
(labelled `My liked trade`) and the walk continues past it.

**Reopen trigger.** The trader asks for the hidden rows to be REMOVED from the report, for the
walk to follow the Points order instead of the visible order, or for the M5 alert bar to
cycle the same way. Gate #124.

## RP - one report id, two views (2026-09-15, WISHLIST 10K)

The measured report is a reader, not a decision maker. `scripts/measured_report.py` computes
the five answers from existing evidence and assigns one id from the stable cells plus `as_of`;
a wall-clock tick cannot make a new report, but a matured cell must. Its published JSON and
Markdown are version siblings: identical evidence leaves the last verified files alone, and
changed evidence writes `_v2` beside `_v1`. A failed publish never costs the nightly digest.

Daily Recap's Review tab does not rebuild numbers on the Qt thread. Its worker reads the
published report, prints the same id and cells, and may read a narration already associated
with that id; absence is `no review yet`, never a model request. Copy is clipboard-only.
Export is the trader's click and writes the brief, payload and manifest to their selected
folder. No route uploads, opens a socket, reaches a detector, score, alert, watchlist, Focus,
review queue, `review_policy.json` or a live store. The brief declares every omission, stays
under 32 KiB UTF-8, and reports its token estimate using the desk's existing 2.5
characters-per-token convention. N3's n-descending narration selection is untouched.

## P3 — a proposal is not a trial (Phase 0.32, 2026-09-16)

The next-test proposal has one owner: `scripts/research_proposal.py`. It receives code-prepared,
bounded report facts, validates every observation against the cited cell and exact report id/hash,
requires the cited-cell subset and each cited source window to be declared, and allows only the
non-authoritative `proposed` / `registered_collecting` statuses,
and stores immutable validated JSON with a restored current-JSON/memo pair on a write failure. The
runner spends a trusted session marker before its one model call; an absent code-owned recipe
allowlist refuses rather than letting model text authorize itself. The selection key is
coverage/readiness/repeated questions/unresolved comparisons/active tests, not an observed return;
the existing N3 `narrated K of N` statement remains visible. The local model is advisory text only:
it may name one test to resolve uncertainty but cannot register, execute, amend or promote a test,
call a recipe optimal, generate code or enter any runner instruction.

The report's entry-quality facts come from versioned flat `entry_quality_window_v1` P8 rows,
published by the owned low-priority warehouse outcome pass from its same completed M5 bars and
declared selectors. They are gross excursion facts labelled `gross_excursion_no_exit`, never a
replacement for `outcome_path`; the session/month-scoped `ResearchStore.read_rows` reader builds
the Packet 2 comparison in memory. A missing/unreadable dataset is one unknown cell and cannot
cost the old outcome build, publish a model proposal, or reach a detector, score, alert or live rule.
When more than one fixed window exists, `30_trading_minutes` is the declared primary by identity;
the other sorted available window names remain metadata and no result value may select the report input.

The Daily Recap Review worker reads the current view only when its current report identity belongs
to the report it already loaded, then the card and clipboard brief render that view while naming the
immutable proposal source separately. It never computes a movement number or reads the AI store on
Qt. A model-free refresh updates only the current trial/no-trigger/missing facts and its memo; a
missing prior proposal, model outage, validation refusal or failed publish leaves deterministic
facts and the last good pair intact. Setup Tracker may route that exact published display object
into Daily Recap > Review without tracker ranking, Qt I/O, a store read or a model call. The proposal is therefore
not permission to change a detector, score, alert, ranking, watchlist, Focus, journal or policy.

## Day Review - one page, no machine rows, a per-session index (2026-09-17, packet TJ-1)

**The trader's words.** "daily recap and market journal feel less than ideal ... I feel
like currently it's overcomplicated ... Market journal just sucks, it has information but
it really should be compacting the days ... rename paste weekly forecast to paste daily
forecast, that's where I paste the output from my scheduled chatgpt prompt ... i don't need
to see the SPY auto modes pasted in there ... there's just too much shit in these tabs and
it's laggy as all hell. this should be a simple 'what worked what didn't and what was your
process'. ... if you think it's better to combine daily recap and market journal that's
fine with me." Answers: `docs/decisions/0021-trader-journal-consolidation.md`.

**What was measured, not assumed.** On a staged copy of the live home folder at 1640x980,
three repeats (`scripts/ui/desk_bench.py`, which TJ-1 taught to build both sides of the
swap): `daily_recap.reload` settled in **13,500-16,500 ms across runs** (p50) because
`daily_recap_reader._read_intraday_outcomes` streams the whole 476 MB
`intraday_bounce_outcomes.csv` on every open with the 29 MB horizon CSV behind it, and
every view then filters by session AFTERWARDS. `market_journal.construct` cost 362 ms
(sync p95) and its first show settled 570 ms (p95). On the live desk the desk's OWN
`Auto mode X -> Y` journal rows were **34 of 77**, and the 2026-09-16 nightly narration
read them back as if the trader had written them.

**The three rules the page is built to.**

1. **One read, one payload.** `DayReviewService.read_day` returns one mapping with every
   section in it and the page paints from that and nothing else, on ONE `QThread`. A
   section that owned a read could start one on the Qt thread; a section that owns a
   payload key cannot. Each store is read in its own guard, so one unreadable store costs
   one section and names itself in `payload["error"]` - a blank section with no reason
   reads as "nothing happened", which is a claim nobody measured.
2. **One chart, built on first need.** The retired Market Journal page built FOUR
   `CandleChart`s on the first entry click (299 ms, the G0 baseline). This page constructs
   with ZERO, builds ONE when a payload carries SPY bars, and reuses it. A past session has
   no bars in the running scanner's memory, so it says "chart after the close" rather than
   drawing an empty axis; TJ-2 brings the stored bars.
3. **No machine row, twice over.** `market_journal.is_machine_entry` is the ONE filter and
   it is applied in `MarketJournalService.entries_about` (which is how `daily_story` and
   `theses_for` select their rows), in `market_story.build_daily_story` (defensively, in
   the pure function, so TJ-4's packs cannot forget it) and in
   `market_story_rollups._stories_from_journal` (the nightly packs). The page filters again
   because it is the surface the trader complained about. A Trade Mentor answer is NOT a
   machine row: the trader wrote every word of it and the desk only chose the moment.
   **Nothing is deleted** - the ledger is append-only and the 34 rows are still on disk.

**The flip.** `MainWindow._record_auto_mode_flip` keeps its name and its caller and writes
one Auto Pilot log line through the new public `AutopilotService.log`, which forwards to
`_log` - still the one writer of the deque, the file, the `logging` line and `logMessage`.
The sentence "Written by the desk, not the trader" is gone because it was written for a
JOURNAL reader; in the Auto Pilot log every line is the desk's.
`market_journal_capture.REASON_MODE_FLIP` stays defined: the old sidecars carry it.

**The index.** `scripts/day_review_index.py` writes
`DAY_REVIEW_DIR/sessions/<date>/outcomes.json` holding exactly the `_Store`s
`read_session` would have built for that session: the rows its views consult (latest append
per `event_id`, in the streaming reader's own order), the **full-file** `SourceCoverage`
carried forward unchanged, and `raw_rows_by_session`. Storing the kept-row count as
coverage instead would quietly relabel a 476 MB file as a 40-row one, so the equality test
is on the whole `RecapSession` dataclass, coverage included. `read_session(index=...)` uses a
valid index for THIS session and THIS lookback and otherwise STREAMS - another session,
another window, a corrupt file, a revival that fails or a store the index does not carry all
read the stores, because a cache may never change the answer. `is_stale` is the one
staleness rule: only a horizon row that had not matured can change, so a pending index is
stale once the session it was waiting for has CLOSED, an index built before its own session
closed is always stale (that file is still being appended to), and an index with nothing
pending is never stale. The post-close tick builds it once, through ONE named seam,
`DayReviewService.build_index_for`.

**Two things that may NEVER happen on the Qt thread, both found by review (2026-09-17).**
First, the post-close index build: it was called straight from the 60-second timer slot and
froze the desk for **22.8 seconds** while it streamed the big stores. It runs on
`_IndexBuildWorker` now - single-flight, the page says "Building <date>'s index in the
background…", the slot returns in 0.2 ms, and the page repaints if the finished index is the
session on screen. Second, `alert_center.journal_chart_bars`: it LOOKS like a cache read and
is not one - it mutates `_m5_bar_dicts` and arms a `QTimer.singleShot`, and a `singleShot`
armed from a thread with no event loop never fires, which latched
`_d1_prefetch_flush_armed` True and killed D1 prefetch for the rest of the session. So the
Qt-thread slot that starts a read calls it (once, and only for a session that has not
closed) and hands the bars into the worker's inputs; `DayReviewService` holds no bars reader
and `read_day` takes `spy_m5_bars` as an INPUT. `MainWindow` hands the accessor to the PAGE,
never to the service, and a source-level test says so. The tests pin the THREAD IDs, because
that is what was wrong: `tests/test_tj1_day_review_post_close_index.py` and
`tests/test_tj1_day_review_bars_on_the_qt_thread.py`.

**What an index costs the home folder.** One is 22.7 MB, it lives in the SHARED home, and
it is rebuildable - so three rules bound the churn (`write_index`): a session that has NOT
CLOSED is never indexed (its stores are still being appended to and `is_stale` would refuse
the file anyway, so today's page streams until the post-close build), an index whose content
is UNCHANGED is not rewritten (two builds of a finished session differ only in `built_at`,
and 22.7 MB is not worth a timestamp), and the folder is pruned to the newest
`KEEP_SESSIONS` (40) on each write. `is_stale` also compares a stored `sources_stamp` -
`(size, mtime_ns)` per indexed store - because a warehouse recompute REWRITES those CSVs and
can change the rows of a session that closed weeks ago, which no clause about pending
horizons would ever notice; the stamp is only consulted when the caller passes `sources`, so
a clock-only question, and an index written before the clause existed, are answered as
before.

**An APPEND is not a REWRITE, and the scope is a NAME, not a session.** The stamp's first cut
invalidated on any change, and `intraday_bounce_outcomes.csv` is appended to all day by the
M5 scanner: one appended row for another session turned a 609 ms warm open into **12,124 ms**
and rewrote the 22 MB index with nothing on the page different (reviewer, round 2). So a
mismatch is READ, not assumed (`stamp_verdict`): a file that only GREW has its appended TAIL
parsed - seek to the stored size, re-attach the header line so the rows parse against the
column NAMES the store declares - and the index is stale only if an appended row is one it
would have KEPT. A file that SHRANK or changed at the SAME SIZE is a rewrite and rebuilds; so
does a bare `os.utime`, which a stamp cannot tell from a same-size rewrite and which is rare
enough to pay for one rebuild. Anything the tail cannot ANSWER - a row whose session or
symbol will not read, a boundary that is not a line end (the stored size was taken
mid-append), a header-less or unreadable file - rebuilds: uncertainty rebuilds, it never
assumes. When the growth is out of scope the 22 MB body is left alone and the new stamp goes
into a few hundred bytes beside it (`stamp.json`, overlaid by `read_index`, ignored when it
names another session, cleared by the next real body write), so the following open compares
sizes instead of reading the same tail again. The tail is read WHOLE: after a week unopened
that is a week of appends, tens of MB, which is still two orders of magnitude below the
476 MB stream and happens once.

**The scope is `(session, symbol)` because a session-only scope had no live benefit at all**
(reviewer, round 3). Every recent index carries target sessions running weeks forward - six
live indexes from 2026-08-28 to 2026-09-17 all hold 2026-09-18, with targets out to
2026-10-01 - so `trade_date = today` appends always landed on `rebuild`. `_index_scope` now
answers with the SESSIONS where any name counts (the selected session and its lookback
window, which the day's own views read whole) and the `(session, symbol)` PAIRS its own
observations reference; side is deliberately not in the key, because a blank side matches
either.

**What that buys, measured on the staged home** (index of 2026-09-17, one appended intraday
row dated 2026-09-18): an append for a name the index does NOT reference is decided in
**28.3 ms**, the page opens in **502 ms** and the body is untouched; an append for a name it
DOES reference is decided in 24.2 ms and rebuilds - **11,380 ms**, on the read worker with the
Qt thread at 0.17 ms, the page still showing what it had. **And the honest limit: that index
references 1,198 names for 2026-09-18 - effectively the whole scanned universe - so DURING a
session almost every append rebuilds.** The rule keeps the page warm after the close and
outside trading hours, which is when the trader reads it, and it costs 28 ms to find out.
Narrowing further means updating one swing row's `first_favorable_pct` in place instead of
rebuilding, which is TJ-2's question, not this one.
`tests/test_tj1_day_review_index_tail.py` holds the twenty-eight cases.

**How wide the index is, and the line between fast and fresh.** The first cut covered the
two biggest stores and left an indexed read at 2,217 ms, which did not meet the gate's
"under one second". Measured store by store on the staged home (2026-09-17): intraday
476 MB / 4,995 ms, horizon 31 MB / 2,704 ms, **tier 14 MB / 944 ms**, **human-focus 1.0 MB /
164 ms**, then pick-feedback 0.40 MB / 36 ms, review-events 0.27 MB / 15 ms, preference
0.44 MB / 8 ms, annotations 0.35 MB / 4 ms and four more at about 2 ms together. So
`INDEXED_SOURCE_SPECS` covers the FOUR stores over a megabyte, and the six small ones are
read LIVE on every open - a freshness rule, not an oversight: a veto, a note, a favorite or a
staged pick from a minute ago has to be on the page. `alert_review_events.jsonl` and
`preference_trade_outcomes.csv` are deliberately OUT even though the follow-up packet named
them, and the reason is worth keeping: both are rewritten as the trader and the journal move
(the M5 click-away verdict lands in the first; the second is regenerated as trades arrive),
an index carries no signal for a REWRITE, and indexing them would buy 23 ms. The 10.5 MB
figure once attributed to review-events was a MEASUREMENT ERROR: that is the
`alert_review_events/` segment directory, which `read_session` never opens - it reads the
single 0.27 MB root file. Indexing tier and human-focus cannot change any answer at all,
because `read_session` opens both for their COVERAGE LINE alone and no view reads a row of
either.

**Two lookups that were walks.** With the stores indexed, the whole remaining cost was
compute: `_d1_horizon_row` walked the entire horizon slice once per D1 decision (117
decisions over 13,700 rows - 1.1 s of profiled time, the single biggest item left) and
`_outcome_for` walked every outcome of the session per call (3,281 calls, 274 ms). Both are
now asked through a dict built once per read (`_horizon_rows_by_name`, `_outcomes_by_name`),
and **neither dict is keyed on SIDE or horizon**: a row with a blank side matches either
request, so those rules stay inside the lookup, applied to one name's handful of rows in
file order, and the FIRST match is still the row the walk returned.
`tests/test_tj1_day_review_index_wide.py` checks both against a brute-force reference walk
over rows built to make the two able to disagree. That took the indexed read from 812 ms to
265 ms and the page's whole read to 396 ms.

**The forecast.** The button reads "Paste daily forecast..." and the dialog asks for the text,
the SESSION it is about and the source model. `MarketJournalService.import_daily_forecast`
files the entry against `target_session` and `market_thesis.record_forecast` records it
beside `target_week`, which is kept and kept separate - a weekly forecast is not a daily one
and writing one into the other's field would make both unreadable. A second paste for the
same session SUPERSEDES the first, so the page shows one brief and both rows stay on disk.
`import_weekly_forecast` survives one release as a deprecated alias.
`scripts/forecast_brief.py` is the view over the text: heading match only,
case-insensitive, no model, nothing fetched, and anything the document does not say stays
empty. Two readings worth writing down, both from the brief the trader pasted on
2026-09-17: `~8/10 through tomorrow, then potentially falling toward 6-7/10` is THREE
scores (8, 6, 7) because a range scores both ends, and `around September 29-30` on a
Turbulence line is NONE, because it carries no `/10`. The ranked-signals line is split on
the arrow in the order written - a set would lose the ranking, which is the whole content of
that line.

**Move, not delete.** The Daily Recap's Review tab is a **Measured report** section at the
foot of Research > Results, with the same cells, the same `report_id`, the same
`review_cells` parity seam and the same `ReadWorker`; `latest_published(root)` with no
session named answers "the newest session that has one", which is the right question for a
page with no session picker. Its read waits for the page's FIRST SHOW, not the constructor,
because that is G7.1's rule for this page - the desk builds nine Research children at
startup and each loads when its tab is opened. The first cut read it in `__init__`, which
gave the desk a thread for a section nobody had looked at and could outlive a panel that was
only `deleteLater`-ed; `tests/test_tj1_measured_report_load_rule.py` pins WHO STARTS the
read rather than when it lands, because on a worker the second question is a race even when
the answer is wrong. The Staged picks table and its verb are on the **Auto Pilot**
page under the log, keeping `focusAddRequested` so the add is still performed by
`FocusService`, the store's own owner - AWAY still STAGES and never adopts, the page still
only ASKS, and the R2 adoption gate is SHOWN at click time and never enforced.
`SetupTrackerPanel.open_entry_quality_review` learned that its host may be a SECTION rather
than a tab: it renders, then hops the tab strip only if the host has one.

**What is gone from the trader's screen** (plan.md 12.2): the environment timeline, "What
the desk measured that session", the calendar overlay, the thesis drafting pane and "Save
interpretation", the four capture panes and the five Daily Recap tabs. Every store stays;
`market_journal_panel.py` and `daily_recap_panel.py` stay on disk, unregistered and no
longer constructed, until TJ-8 deletes them - which is why the WS-RP and Sol proposal tests
that construct the recap class directly still pass as written.

**What it cost, measured the same way each time** (`scripts/ui/desk_bench.py`, staged home,
1640x980, three repeats). Retired page: `market_journal` construct 362 ms sync p95, first
show 570 ms settle p95, entry click 121 ms; `daily_recap.reload` **13,500-16,500 ms across runs** settle p50.
Day Review, **cold and warm stated separately because they differ by 15x** (the first
handoff printed 1,099 ms for the cold open, which was a warm bench figure and wrong -
reviewer, 2026-09-17): construct 4-20 ms; **the Qt thread costs 0.2 ms either way** (a
`reload` only starts a worker). COLD, with no index on disk: the page paints after
**12,549 ms**, because that read streams the four big stores and builds the index behind
it. WARM: the page paints after **607-677 ms**, `day_review.reload` settles at **820 ms p50
/ 834 ms p95** on the bench (an earlier 550 ms p50 was measured in the process that had just
written the index; 820 ms is the conservative number), entry click 121 ms settle / 0.6 ms
sync. The POST-CLOSE tick: **slot 0.2 ms, index lands 11.0 s later on its worker** - it was
22.8 s of frozen desk when the slot did the work itself. One index is 22.7 MB; reading it
costs 92 ms.

**TJ-2A session bars (2026-09-18).** After the post-close index build, its same worker
fetches regular-hours Yahoo M5 bars for all decided/traded names plus SPY, QQQ, IWM and
VXX in batches of 50. `day_review_bars.py` keeps completed bars only and atomically stores
them under `DAY_REVIEW_DIR/bars/<session>.parquet`; a failed symbol is absent. Closed-day
reads take SPY from this durable file, while today's tape remains the Qt-thread cache handoff.
A missing past file starts one backfill worker and never fetches the in-progress session.

**TJ-2B walk-away (2026-09-18).** `walkaway_day.build` is pure: it reads the worker's durable rows and bars only. A source duplicate is one decision but two decision times remain two rows; claims replay by key so a later drop is historical context, never a deletion. A later matched trade moves a like to left-early, and absent exit bars, horizons, or maturity are named states rather than zeros.

**Layout: two columns, and tables that fill the width** (TJ-1L, 2026-09-18). The trader, the
first time he read the page on a 3800 px screen: *"there's a lot of empty space horizontally
that's not being efficiently used"* - and, offered three shapes, he chose two columns. The
page is still ONE `QScrollArea` and still builds ONE chart on first need; what changed is
where things sit. Row 1 is the session picker, full width. Row 2 is a horizontal `QSplitter`
named `DayReviewColumns`, default 55/45, restored at construction and saved per machine on
the drag through `ui.panels.desk_layout`'s three calls (`qt_day_review_columns_v1`) - the
same seam the desk's D1 column split uses, because a split the trader drags and finds moved
next morning is worse than no split. LEFT: *What happened* with *Open theses* directly UNDER
it (not beside it - the theses are a short list ABOUT the story) and then *SPY, this session*
with a 320 px floor, taking whatever slack the column has, because the chart is the one thing
on this page that turns width into information. RIGHT: the entries list over the reader in
their own 60/40 vertical splitter (`qt_day_review_said_split_v1`, a 300 px floor so the 40%
half clears the reader's own 90 px minimum and the preset IS the preset), then *New entry*
spanning the column, its Timeframe / Save / Paste row, the "filed under the session" note and
*External forecast* collapsed to three lines. **Every box in that column spans the column**:
the G3 100-character cap that used to hold the reader and the forecast at a readable measure
is what put the empty space back once they sat in a 45% column - measured at 3800x2000 both
stopped at about 420 px of an 890 px column while *New entry* under them ran the full width -
so `refresh_reader_measure` keeps its name and its `MainWindow._apply_scaled_metrics` caller
and now CLEARS the cap and sets `Expanding` instead of setting one. The column the trader
drags is the measure. (The Market Journal's own reader keeps the G3 cap; that page is a
different shape and `tests/test_g3_market_journal_reader.py` still pins it.) Row 3 is
*Walk-away* as a 2 x 2 grid of equal
COLUMNS with rows that fit their content: the one real table top-left and TJ-2's three
populations as small titled frames, top-aligned, one title line and one note line each -
three empty tables padded to a table's height would read as three tables that failed to load.
Row 4 is *What you traded* beside *Ideas from the desk's AI*. There is no trailing stretch on
the page: the slack belongs to the columns, and inside them to the chart.

**Why the tables stopped clipping their own headers.** "Against me first %" printed as "ainst
me first" and "After the decision %" as "r the decisio": the shared width rule
(`ui.widgets.data_table.apply_width_rule`) clamps every measured column to `MAX_COLUMN_WIDTH`
(260 px), and under the desk theme that header hints 273 px - a header is CENTRED, so a clip
shows at both ends and there is no ellipsis to warn anyone. This page's two tables now use
`_fill_the_width` instead: every column but the last is `ResizeToContents` (which is never
narrower than the header's own hint) and the last section stretches, so the table fills its
cell on a 3800 px screen instead of ending in the middle of it. It is set ONCE at
construction - `ResizeToContents` re-measures itself when the rows change - so a repaint
costs the rows and nothing else, and the measurement is bounded by the same
`MEASURE_PRECISION_ROWS` cap the shared rule uses. The shared rule is untouched and still
serves every other table on the desk; this page opts out because its tables hold one day.

**What the shape cost** (`desk_bench.py`, staged home, 1900x1000, seven repeats): page
construct settle p50 **33.6 -> 53.3 ms** and the whole panel 23.6 -> 38.4 ms in a direct
profile, all of it in `_build_layout` (two splitters and about fifteen more widgets, once per
desk start); first show settle p50 **900 -> 864 ms**, `day_review.reload` **876 -> 869 ms**,
entry click **122 ms** either way. The two numbers the trader feels - opening the page and
clicking an entry - did not move.

**Tests:** `tests/test_tj1l_day_review_layout.py` (21, of which 17 failed and 1 errored on
the pre-change panel) pins the shape: one horizontal splitter with exactly two widgets and
what belongs in each, the theses above the chart, the 60/40 reader split, the four grid cells
in their stated positions, a placeholder that is a frame with a title and a note, the two
halves of row 4, `stretchLastSection` on every table, the split ratio round-tripping through
the saved setting, and - re-asserted here because a layout rewrite is exactly the change that
could quietly build two - ONE `CandleChart`, built on first need and reused. The three that
pass either way are named as guards in the file.

**Tests:** the tester's `tests/test_tj1_machine_rows.py`,
`tests/test_tj1_page_specs.py`, `tests/test_tj1_day_review_page.py`,
`tests/test_tj1_day_review_index.py`, `tests/test_tj1_forecast_brief.py` and
`tests/test_tj1_moves.py` (104 tests, 96 red at the tester's commit), plus three builder
files for the seams none of them pinned, each proven red first:
`tests/test_tj1_day_review_post_close_index.py` (the one named build seam),
`tests/test_tj1_day_review_index_staleness.py` (the two edges of the staleness rule, found
by the bench) and `tests/test_tj1_day_review_index_wide.py` (the widened index's equality,
its all-or-nothing revival, and the two rewritten lookups against a reference walk).

## Day Review - walk-away v2: two rulers, a real-miss rule, and a base rate (2026-09-19, packet TJ-11)

The long form behind the CLAUDE.md rules *"A real miss is a RULE, and the session a
decision belongs to is an ADDITIVE field"* and *"A D1 call gets a D1 ruler, and a miss is
read against a base rate."* Trader, 2026-09-19: *"if they said no to a bunch of stocks
that went on to have great moves that day or the next day, then I want to know about it.
… I want what I missed to be very apparent. I want what I did well with to also be very
apparent."* Decision 0021 answers 22 and 24; plan.md §12.4 TJ-11. Branch
`claude/tj11-walkaway-v2`, tip `a744bec1`, merged `a89ec7d5`.

**What was measured** on the live stores, read-only, 2026-09-18/19: session 2026-09-17
carried **146 D1 decisions and 94 M5 ones, 240 in all** - the packet's "6 M5" was wrong -
and TJ-2B graded every one of them on that session's five-minute tape, which is the wrong
ruler for 146 of them. The 18 D1 calls made on Friday evening at **21:04-21:07 Pacific**
were stamped `session_date = 2026-09-19`, a **Saturday**: a session that never happened
and a date Day Review could never show. 50 trades since 08-04: 8 options, and 15 of 37
closed trades held past five sessions.

**Two rulers, never mixed.**
* An **M5 decision** is measured on the session's own tape, from the OPEN of the first
  completed bar after the stamp (TJ-2B's rule, unchanged).
* A **D1 decision** is measured on DAILY bars (`chart_snapshot.load_d1_bars`, the durable
  parquet store, off the Qt thread) from the **CLOSE of the session it was made in**, over
  1, 3 and 5 exchange sessions. Until the five-session horizon closes the row reads
  `pending <date>` and every move is blank - **never zero**. The horizon dates come from
  the exchange calendar, because the live horizon-outcomes file has no `maturity_date`
  column.
* ATR(14) is **point-in-time**: the bars up to and including the decision's own session.
  An ATR that included the move being measured would shrink the yardstick for that move. A
  missing ATR leaves the ATR columns empty and the percent columns alone.

**`REAL_MISS_V1` (`scripts/real_miss.py`) is a rule, not a glance.** A real run reached
`RUN_ATR` (1.0) x ATR in the decision's favour BEFORE it went `ADVERSE_ATR` (0.5) x ATR
against. Inside ONE bar the order of the two extremes is unknown, so the **adverse one is
taken first** - a bar that touches both is `no_run`. A missing ATR, or no completed bar
after the stamp, is `unmeasured:<reason>`, never `no_run` and never zero. The module is
import-light on purpose: TJ-15's nightly slot calls the SAME function under
`requirements-core.txt`, so there is one rule and not two. A `no_run` over an UNFINISHED
window is not a finding, so a D1 verdict is `unmeasured:horizon_open` until the window
closes - while a `run`, once reached, cannot be taken back.

**Against you first** is the worst adverse extreme up to and INCLUDING the bar that made
the best favourable one - what the trader was up against before the move, not the
give-back after it. It is never positive.

**`market_calendar.decision_session(stamp)`** is the one session-stamp seam. **CHANGED
2026-09-19 by TJ-11F** - it answered *"the session the stamp falls in, else the NEXT
exchange session; evening, weekend and holiday all map forward"* for one day, and now
answers the **session whose New York calendar date the stamp falls on** when that date is
a session day - pre-market, in-session and after the close all stay on that day - **else
the most recent PRIOR session**. Evening, weekend and holiday map BACK. The close is no
longer a boundary, so nothing here needs `session_close`; `next_session` stays as its own
helper, called by `walkaway_day` and `day_review_service._stamped_dates_for`. Zones are
converted with `astimezone`, never stripped; a naive stamp is read as market-local, and a
full ISO TIMESTAMP in text is parsed as a MOMENT and never truncated to its first ten
characters, so an aware stamp in another zone cannot answer the wrong day.

**And the session a decision belongs to is an ADDITIVE field, never a new meaning for an
old one** (reviewer NO-GO at `21ed0eb6`, closed at `a744bec1`). `session_date` is the join
every live reader uses - `review_learning`'s veto cohort behind `review_policy.json`, the
three cohort graders, `daily_recap_reader._decisions`, and `pick_feedback`, which feeds the
setups-table hide, the chart-cycling skip and the review queue's "Reviewed today" mark.
TJ-11's first build moved both the written value and `load_annotations`' DEFAULT join, and
the reviewer reproduced it on a copy of the live stores: on Monday 2026-09-21 the branch
would have hidden **12 setups rows where base hid none** and marked **18 names
Reviewed-today**, and the writer change would have moved Friday-evening vetoes into
`review_learning`'s veto cohort, the cohort graders and the recap.

The lead's binding design: **`session_date` keeps EXACTLY its base meaning and value for
every writer and reader.** The decision's session is a NEW ADDITIVE key,
`decision_session` (`store.DECISION_SESSION_FIELD`), written on NEW rows only, empty rather
than guessed when the calendar cannot answer, **never backfilled** - an old row simply
lacks it. Only TJ-11's readers use it, taking the stored value **only when the row also
carries `decision_session_rule: "judged_session_v2"`** and otherwise mapping the row's own
stamp back through the calendar (`store.row_decision_session`, `walkaway_day._row_session`;
the marker and the backward direction are TJ-11F's, 2026-09-19 - this read took the stored
value whenever it was present and mapped the stamp FORWARD when it was not),
and `load_annotations(..., by_decision_session=True)` is the explicit opt-in that nothing
outside TJ-11 passes. A cohort row is built from named fields, so the extra key adds no
column to any grader. Parity with base is PINNED by tests for `pick_feedback`,
`review_learning`'s join, the three cohort graders and `daily_recap_reader._decisions`.

**The Saturday-stamped picks were left where they are** (lead decision (a)). The 18
annotation rows carrying 2026-09-19 became **12 veto-cohort and 6 like-cohort picks** with
a non-session `trade_date` and a Monday-close ruler, and **zero outcome rows**. The graders
and those rows are deliberately untouched: nothing on the live desk changed, and only
TJ-11's readers map them at all (forward then; BACK since TJ-11F, so those 18 rows read on
Friday 2026-09-18).

**The skill line: a miss is always read against a base rate** (decision 0021 answer 22).
Three populations of the SAME scan - liked/claimed, rejected, untouched (shown by the scan,
no verdict from the trader) - PARTITION the scan's rows, each with `n`, `measured`,
`pending`, `runs`, the rate and the ONE Wilson interval (`swing_headline`'s z 1.96, both
ends), cut by side and, where `measured >= MIN_REPORTABLE_N`, by setup family. `n` is the
population (one row per name per session in the window - the label says **scan rows**);
`measured` is the Wilson denominator. Both windows are rendered, each saying its window in
SESSIONS, and every cell prints `n`, `measured` and `pending` ALWAYS, not only when they
differ.

**A POOLED RATE COUNTS A NAME ONLY WHEN ITS HORIZON HAS CLOSED** - runs and no-runs alike
(the second reviewer NO-GO). The first build kept an early `run` inside an open
five-session horizon while holding back that window's no-runs as `unmeasured:horizon_open`,
so the numerator could grow where the denominator could not and every open-horizon cell
read **100% by construction**: the lately LONG rejected cell shipped **34% (30/87)** where
the closed-horizon-only truth was **26% (20/77)**. Open names are in neither half of the
fraction and are printed as their own count, `pending P`; a session whose horizons are all
open reads `measured 0, pending N` and names no rate. The ROW may still show its early run
- a row is a fact about one name, a rate is a claim about a group - and `_D1Reading.pool`
says which bucket a name is in so no reader infers it from a verdict string. Overlapping
intervals are SAID to overlap. Size and name order the view - **no R statistic selects what
is shown** (gate #43).

**One deterministic sentence per table**, e.g. `You vetoed 92. 7 were real misses; 4 share
the reason extended.` Only the VETO vocabulary is counted in the reason clause - a
pick-feedback "not today" is a verdict with free text, not a code the trader chose - and
overlapping codes are never summed (the top code needs at least two rows).

**Instrument-aware rows** (decision 0021 answer 24). An option trade and a position held
past `LONG_HOLD_SESSIONS` (5, counted in EXCHANGE sessions) keep their money figure and
lose "left on the table today"; the row says `not judged here: <reason>` instead of a wrong
number. Assignment is read off the LEGS (IBKR writes `Buy 100 … (Assignment)`); legs that
say nothing leave it `unmeasured`, never "not assigned". Every money line carries its `n`
and reads "too few to call" under `MIN_REPORTABLE_N`.

**What the page costs, and the caps that bound it.** One worker, one payload, lists diff,
no second read. `read_day` hands the builder the session's decisions WITH their reason
code, every decision of the lately window, the session-and-lately scan rows (the
horizon-outcomes read it was ALREADY doing - the 1.1 GB tracker is never opened by a page)
and daily bars. **Daily bars are bounded by SIZE rules, never by a result**: the trader's
own decisions in full, then `EARLIER_SYMBOL_CAP` (150) earlier names and
`UNTOUCHED_SYMBOL_CAP` (150) untouched names in NAME order, `DAILY_BAR_SYMBOL_CAP` (400)
in total, each trimmed to `DAILY_BAR_TAIL` (60) bars, and a symbol this read put into
`chart_snapshot`'s process-wide cache is dropped from it again. Measured 2026-09-19: the
daily-bar read at the caps is **3.37 s and 112 MB**, against a session scan of ~1,100
distinct names that would cost ~5 s and ~340 MB unbounded. `read_day` as a whole measured
**6.8-9.8 s per session on a staged home**. A name past a cap is `unmeasured` and every
cell says `measured K of N`.

**The render rule this packet learned the hard way.** `_fill_the_width` leaves every column
but the last in `ResizeToContents`, and Qt re-measures those columns on EVERY `setItem`
(bounded to 200 rows, through a styled delegate, on the GUI thread). Five tables and ~700
rows spent **90 seconds inside one `processEvents`**. The mode is suspended for the fill
and restored once. Reload's worst single GUI stall on the staged home: **672.9 ms before
TJ-11, 92.4 ms after**, and settle-deadline hits went 2 -> 0.

**Four reviewer advisories, recorded and not repaired.**
* `load_annotations(by_decision_session=True)` and `row_decision_session` have **no
  production caller yet** - the service maps through `_stamped_dates_for`. They exist for
  TJ-15 and are pinned only by tests.
* The stored `decision_session` and the service's tag **DISAGREE for a same-day after-close
  call**: for a Friday 16:30 ET veto the stored field says Monday while the page shows it
  on Friday. TJ-15's nightly slot must pick one of the two deliberately, and say which.
  **CLOSED by TJ-11F (2026-09-19): both say FRIDAY**, pinned by
  `test_the_stored_session_and_the_page_agree_for_an_after_close_call`.
* `market_calendar.session_close` is a flat 16:00 with no early-close modelling. This is
  pre-existing and TJ-11 did not widen it.
* On a fresh day the SESSION skill line honestly reads `measured 0, pending N` and names no
  rate; the LATELY line is the one carrying content. That is the closed-horizon rule
  working, not an empty page.

**Open trader question** (not blocking): should a Friday-evening veto hide its setups row
and count for the cohort on MONDAY? Today it is stamped with New York's Saturday date and
nothing about that changed. **ANSWERED 2026-09-19 for the PAGE only (see TJ-11F below):
the veto reads on FRIDAY. The setups-row hide and the cohort are still driven by
`session_date` and are untouched.**

### TJ-11F - the rule was reversed the day after it shipped (2026-09-19)

Wave 1 went live on the desk at 16:03 PDT. At ~16:20 PDT the trader answered the lead's
TJ-11 question: *"a veto on friday night (after the market close) should not be considered
monday since we have new information then."* That strikes decision 0021 answer 16's last
clause (now its answer 33) and `plan.md` TJ-11 item 5 as TJ-11 had built them earlier the
same day. A decision belongs to the session whose information it JUDGED; the next
session's scan is new information. Branch `claude/tj11f-decision-session`, tip `67143e3e`,
merged `f00ec302` into `lead/p033-integration2`.

**What moved.** `market_calendar.decision_session` maps BACK, not forward. The annotation
writer adds `decision_session_rule` beside `decision_session`.
`day_review_service._stamped_dates_for(session)` is now `session` plus the non-session
dates AFTER it, up to but excluding the next session day - Friday owns its Saturday and
Sunday, Friday 2026-09-04 owns 09-05, 09-06 and Labor Day 09-07, and Monday owns only
itself. The D1 ruler's reference moved with it: an after-close call is measured from the
JUDGED session's close, so the 2026-09-18 evening calls are measured from **Friday's**.

**What did not move.** `session_date` is byte-identical to base for every writer and
reader; `pick_feedback.decisions_today` was reproduced byte-identical to base for 09-18,
09-19 and 09-21. `review_learning`'s veto join, the three cohort graders and
`daily_recap_reader._decisions` stay pinned and green, and the 18 Saturday-stamped cohort
picks are left exactly as they are (lead decision (a)). **Nothing on the live desk outside
Day Review changes**, and `decision_session` has no caller outside Day Review.

**The marker, because a row is never rewritten.** A new row carries
`decision_session_rule: "judged_session_v2"`. A stored session WITHOUT the marker was
computed by the struck forward rule, so every reader ignores it and recomputes from the
row's own stamp; a marked row is believed. It is a schema stamp, not a vocabulary version.
Measured read-only on a COPY of the live annotation store, 2026-09-19: **1,198 rows, 0
carrying `decision_session`, 0 carrying the marker** - the file's last write (2026-09-18
21:07) predates wave 1 going live - so the marker is insurance against rows the running
desk could have written in that window, not a migration.

**Reproduced on that copy.** All 18 Friday-evening calls (12 veto + 6 `like_claim`, filed
21:04-21:07 Pacific and stamped New York's Saturday date) appear on `read_day("2026-09-18")`
- 150 rows = 132 + 18 + 0 - and **none** on `read_day("2026-09-21")`, which holds zero.

**Calendar edges, stated because they are not obvious.** A call over the Labor Day weekend
answers Friday 2026-09-04; a Thanksgiving-Thursday evening answers Wednesday 2026-11-25;
an early-close Friday evening stays on its own Friday; garbage answers `None` rather than
guessing. And 23:00 Pacific on a Sunday is 02:00 Eastern on the Monday - a session date,
before its open - so it answers **Monday** under the struck rule and the new one alike,
the same date the base `session_date` carries. Only a NON-session New York date walks
back, and **0 live rows fall there**.

**Three advisories, recorded and not repaired** (reviewer GO, 2026-09-19).
* The rule marker is **two independent literals** - `store.DECISION_SESSION_RULE` and
  `walkaway_day.DECISION_SESSION_RULE`, both `"judged_session_v2"`. They must move
  together and nothing makes them; this wants ONE imported constant.
* Day Review's "Today - provisional" entry lets the page open on a **NON-session date**. On
  a Saturday, `_stamped_dates_for("2026-09-19")` returns the weekend itself and the service
  tags those in-memory rows `decision_session: 2026-09-19` with the v2 marker, although the
  rule would say 2026-09-18. The rows shown are the same 18 the base build showed on that
  page and **nothing is written to a store**, but the tag is wrong in memory; this wants a
  guard that asks the calendar before tagging.
* `load_annotations(..., by_decision_session=True)` and `row_decision_session` still have
  **no production caller** - the service maps through `_stamped_dates_for`. They exist for
  TJ-15 and are pinned only by tests.

## TJ-10 - the read grader: a verdict is measured, and the band is the lead's number (2026-09-20, packet TJ-10)

The long form behind the CLAUDE.md rule *"A read is a MEASURED row, and a clicked call is
never pooled with a stance we inferred"*. Merged into `lead/p033-integration2` on
2026-09-20 (merge `57b44ca9`, branch tip `2967e4a6`, integration fix `3e52d94f`); `main`
has not been fast-forwarded to it yet and live gate #155 is owed.

Decision 0021 answer 14: *"'Were you right' is a MEASURED row, never a model's opinion."*
`scripts/market_read_grades.py` is that row and it is pure - no thread, no sleep, no clock
of its own, no notifier, **and no model anywhere in the packet**. Everything it may see is
handed in.

**A click wins over the words.** A READ is either a CLICKED prediction
(`market_journal.prediction_of`, the ONE accessor) or a stance EXTRACTED from the trader's
prose (`market_thesis.extract_thesis`), and the two are never pooled - `pooled_accuracy`
raises `PoolingError`, because an inferred stance and a stated one are evidence about
different things. `no_view` is recorded and never graded. `select_read` is the ONE
selection rule: a CLICK always beats an extracted stance of the same timeframe, and
contradictory extracted notes compare with NOTHING (`your notes read both ways (1 up, 1
down) - no single read to compare`, verdict `unmeasured`).

**What is measured.** `rest_of_day` anchors on the OPEN of the first completed M5 bar that
STARTS after the stamp - never the stamp's own bar - and ends at the last completed bar at
or before `session_close`, half days included; the durable tape is Pacific-local and the
stamp UTC, converted with `astimezone` and never `replace(tzinfo=None)`.
`next_5_sessions` anchors on the DECISION session's own daily close
(`market_calendar.decision_session`, so a Friday 21:04 PT call - or a Saturday one -
anchors on FRIDAY's close, the trader's 2026-09-19 judged-session rule), reports
checkpoints at 1, 3 and 5 exchange sessions and takes its VERDICT from the 5-session one;
until then it is `pending <date>`, never zero and never wrong. While the decision session
is still trading the anchor is `None` - a forming close is not a close (follow-up fix
`2967e4a6`, after the reviewer reproduced an anchor of 130.0 taken off a still-open bar).

**The flat band is `FLAT_BAND_ATR = 0.25` of the benchmark's POINT-IN-TIME daily ATR(14),
edge INCLUSIVE, and it is the LEAD's number.** It exists because a directional call on a
day that went nowhere is not a wrong call, and counting it wrong would make a trader who
reads chop correctly look like a coin flip. Truth table: +1.00 ATR with an `up` read is
`right` and `down`/`chop`/`range` are `wrong`; +0.25 and +0.10 ATR are `flat` for `up` or
`down` and `right` for `chop`/`range`; -0.30 ATR with a `down` read is `right`. `flat` sits
IN the accuracy denominator (`n` = right + wrong + flat). **The packet named no number, the
lead chose 0.25 on 2026-09-19, and it is the TRADER's to change - never from one session's
result** (plan.md sec 6). Every row carries `flat_band_rule: "atr_0.25_v1"`, so a later
band supersedes cleanly instead of silently re-grading rows measured under this one.

**A verdict may only ever move UP** (round-1 fix). The reviewer reproduced the defect this
replaces: `regrade_matured` revisited only `pending*` rows, could not measure on this desk,
and DESTROYED the row it failed on - an absent daily store turned a correct
`pending 2026-09-25` into `unmeasured:no_anchor_close` forever, when the true answer was
`right`, +5.0 ATR. `verdict_rank` is now the rule (measured > pending > unmeasured-for-a-
data-reason > final): a new row is appended only when the rank RISES, an `unmeasured`
result never supersedes a `pending` one, a data-unmeasured row is revisited on EVERY later
run, a measured verdict is FINAL (contradicting bars later write nothing),
`unmeasured:not_a_call` is final because no bar turns a `No view` into a call, a re-grade
supersedes with a NEW row carrying the ORIGINAL context forward, and the file only grows.
`DayReviewService.build_reads_for` keeps the same rule. **ONE loader set serves the page
and the night** - `daily_bars_for_symbol` (the durable store, then the machine-local daily
cache), `session_bars_for` (TJ-2A's tape) and `atr_for_session` - because a page and a
night that read different stores grade different markets, which is exactly what happened
before the fix round.

**Every surface names its SOURCE, in plain words** (round-1 fix). The page had presented an
EXTRACTED stance as the trader's own call - "your D1 read is up" on 2026-09-17, inferred
from a hedged sentence while another note that day read down. A click now reads
`your call: down (clicked 07:02)`, an extraction `we read your note as up`, on every
congruence line, every entry label and the reader's chip (`read_phrase`, `_read_chip`).

**Every gradable grade carries its point-in-time context, and the store is STRICTER for a
CLICK** (TJ-16 item 1, shipped here so no graded click is ever stored without one; the
strict half is the lead's decision on the reviewer's recommendation). The market half is
`trade_mentor_context_v2` - the entry's own stored block when it has one, else a rebuild
through `trade_mentor_context.internals_at` and through nothing else; there is never a
second builder, and `internals_bars_at` now has its first production caller in
`build_reads_for` (TJ-14A item 6). `append_grades` refuses a gradable grade with a BLANK
context, and refuses a gradable CLICKED grade that carries only the named absence
`CONTEXT_UNMEASURED` (`ContextMissingError`) - an extracted row and a superseding re-grade
may carry it. When `context_for` fails for a clicked read the grade is HELD BACK that pass,
the payload row says `grader_gap: "context_unbuildable: ..."`, and the next pass retries:
the trader still sees the verdict and the reason, rather than a stated call degraded to a
context-less row forever. **Nothing counts the clicks held back that way** - an advisory,
not a defect.

**`grader_gap` now EXISTS on every grade row** (`""` when nothing was missing, else the
measurement gap's reason, or `context_unbuildable: <error>`; deliberately NOT set for
`not_a_call`, which is a complete answer). It is a field, never a signal - nothing reads it
yet, and TJ-14B's dormant `grader_gap` Mentor kind can be woken against it by a later
packet.

**Congruence is printed, never pushed** (answer 15), and it is keyed on TIMEFRAME.
`CONGRUENCE_KINDS` is the three-line D1 spine - the read against the desk's D1 label,
against the session's like/claim side mix, and against the bias of its fills (from the
LEGS: a LONG option is never a bullish setup). `CONGRUENCE_M5_KIND`
(`m5_picks_side_mix`) is APPENDED beside it whenever the session has a rest-of-day read,
pairing it with that session's M5 likes (`not today` counted apart, never folded into the
side mix) and never with D1 picks: TJ-15 measured about half the trader's reviewed
decisions as M5, so a D1 view compared with a mostly-M5 crowd answers the wrong question.
Each line names its `n`, its timeframe and its source ids, names the MISSING side rather
than reading an absence as agreement, and a label with no direction (`compressed`,
`mixed`, `unknown`) is never agreement. **An under-floor side mix is never a verdict**
(follow-up `2967e4a6`): `VERDICT_TOO_FEW = "too_few"` when `n < MIN_REPORTABLE_N`, with the
counts and the floor note kept in the text - live 2026-09-17 had printed `disagrees` off 2
M5 likes, and a chip or a later pooling packet keys on the verdict, not on the words beside
it. On a both-ways or no-read day each line says its OWN content FIRST and the reason there
is nothing to compare it with second. Reviewer's note, recorded not built:
`MIN_REPORTABLE_N` is 30, the "name a winner" floor, so the M5 side-mix line will read
`too few to call` on most days; a smaller NAMED floor for a side mix would be the trader's
or the lead's number. **The congruence D1 label is the SAME session's label** (what the
desk read of the day being reviewed) while the CONTEXT block's `d1_environment` is the
PRIOR session's (the newest one the trader could have seen at the stamp) - two different
questions, deliberately not one field.

**The nightly slot.** `scripts/ai_jobs/read_grades_mature.py` registers `read_grades_mature`
- deterministic, `uses_model=False`, `max_attempts=3`, `reserve_minutes=2.0` - directly
after `theta_pick_grading` and before `miss_contrast` / `market_story_rollups` /
`measured_report`, so it stays INSIDE `_deterministic_stage` and runs on the weeknight,
Saturday AND Sunday slates (a slot appended after `_STAGE_ONE_LAST_SLOT` silently leaves
the Sunday slate however deterministic it is). A missing reads directory, an unreadable
ledger or a missing daily store each give an `ok` row with a reason in under a second and
never raise; the slot is idempotent across the task's 30-minute re-firings; the only file
it writes is `DAY_REVIEW_READS_DIR/<session>.jsonl`. Decision 0018 carries the addendum.
At integration the lead added the slot to TJ-13B's two weeknight-slate pins, and moved the
49-note text fixture to `tests/fixtures/day_review/tj10_live_notes.json` because every
`*.json` directly under `tests/fixtures` must carry the Milestone-3 market-data provenance
contract - the full suite caught it (`3e52d94f`).

**What the desk can measure today** (both reviews ran read-only on STAGED COPIES of the
live stores, 2026-09-20). **0 clicks exist yet** - the click card only went live on
2026-09-20 - and there are **7 extracted reads on each of 09-17 and 09-18**. The vocabulary
bump (`market_thesis.EXTRACTOR_VERSION` = `market_thesis_vocab_v2`: whole-token trend words,
with `rejecting`/`rejected` SCOPED to the phrase that means a level rejected them) moves the
49 live notes from **18 to 20** read as a direction with **no reversal** (23 -> 22
`unstated`); 12 of the unstated fire a bullish AND a bearish word, which no vocabulary can
fix - words are context, and the CLICK is what gets graded. Congruence as the trader will
read it, re-counted exactly by the reviewer from `_decisions` + `claimed_picks`: on
**2026-09-18**, `we read your note as up; 62 of 99 D1 likes and claims were SHORT (long 37,
short 62)` - disagrees; on **2026-09-17** the notes read both ways and 59 of 90 D1 likes
were LONG. `d1_environment.jsonl` holds 15 rows over five sessions and none for 09-18, so
the desk-label line honestly reads `unmeasured` on most days; the fills line reads `the legs
could not call` for the trader's sold puts until they run TJ-9Q's `--apply`.

**Two live-desk facts, stated carefully.** (a) `C:\TradingBotData\data\daily_bars\SPY.parquet`
EXISTS and holds 189 rows through 2026-09-18, yet `chart_snapshot.load_d1_bars` answers
EMPTY for SPY, QQQ, IWM and VXX on this desk for its OWN reason, so far unexplained - a
follow-up question, not a TJ-10 defect. Every benchmark daily bar in the staged runs came
from the machine-local cache `%LOCALAPPDATA%\TradingBotV3\machine_cache\daily_bars` (1,993
symbols). It is NOT true that the home folder has no `daily_bars` directory. (b) The live
home has no `day_review\bars\` folder at all yet (only `day_review\sessions\<date>\
outcomes.json`), so every rest-of-day read is honestly
`unmeasured:no_completed_bar_after_the_stamp` until the first post-close tick writes a tape
(TJ-2A's `build_session_bars_for`) or a back-fill does - which is why gate #155 REQUIRES
the tape to exist before it can be judged.

**Advisories recorded, not repaired.** Nothing counts the clicks held back for
`context_unbuildable`. `baseline_reads` (`always_up`, `same_as_the_last_hour`,
`with_the_d1_environment`) needs GRADE rows to score, so TJ-12 / TJ-16 must feed it those.
The congruence D1 label and the context block's D1 environment are deliberately different
sessions (above). `internals_bars_at` is now called in production.

**The re-review cost the desk a cache file, and the rule changed** (2026-09-20). A
reviewer's scratch script outside pytest set `TRADINGBOTV3_DATA_DIR` but not `LOCALAPPDATA`
- and that variable does NOT move `project_paths.CACHE_DIR`,
`%LOCALAPPDATA%\TradingBotV3\machine_cache` - so the script overwrote the desk's live
`machine_cache\daily_bars\SPY.csv` with a 6-row fake while the desk was down; it was
restored from the durable parquet and the lead verified 189 of 189 rows identical. The rule
in CLAUDE.md "Working agreement for agents" and in `docs/AGENT_TEAM.md` now says BOTH: a
script outside pytest sets `TRADINGBOTV3_DATA_DIR` AND `LOCALAPPDATA` to scratch
directories BEFORE importing anything under `scripts/`, and aborts if ANY `project_paths`
root resolves under `C:\TradingBotData`, the DAS or the real `%LOCALAPPDATA%\TradingBotV3`.
This is the second incident of its kind after 2026-09-05's live-tracker overwrite; the
long form of that one is in `docs/AGENT_TEAM.md` "Rules that exist because something broke".

## TJ-4 - the day pack and the night's voice (2026-09-20, packet TJ-4)

The long form behind the CLAUDE.md rule *"The night narrates the day, never grades it, and
sweeps what the trader queued."* Branch `claude/tj4-day-story` (tip `6c1e2506`), merged
`d929e34f` into `lead/p033-integration2`; `main` has not been fast-forwarded to it and live
gate #148 is owed.

The trader's own words, 2026-09-19, on what the whole program is for: a bot that *"takes in
what I do and think … mathematically deduces what parts of my thinking are profitable and
unprofitable"*. The whole of TJ-4 rests on the amendment that followed: **the model narrates
verdicts, it never makes them.** Five things make that true rather than hoped for.

**1. The pack is the only thing the night may see, and it is deterministic.**
`scripts/day_review_pack.py` is PURE - it opens no store and has no clock of its own; the
caller hands it the day it has already read. It is built by the post-close tick on the Day
Review worker AFTER `build_reads_for`, because the pack's `reads` section IS what TJ-10's
grader just wrote. Building it a second time hours later produces a byte-identical pack,
which is what makes the `inputs_hash` skip honest: the night is not paid again for a session
that has not moved. The clock lives in `built_at`, outside the hash. Twelve `SECTIONS` after
the 2026-09-19 amendment, with `report_card` (TJ-12) and `mood` (TJ-7) present and empty as
hooks. The pack sits INSIDE `day_review_index._prune`'s 40-session delete path, which is safe
only because it is rebuildable - the hash-stability test is what proves it, and the write
seam says so in a comment.

**2. An observation is not a call.** TJ-14A split what the trader SAW from what they CLICKED
at the writer; the pack keeps them as two items with two ids, an empty observation emits no
item, and the night rejects any output that grades an `observation`. A machine row and the
pasted forecast never reach `trader_said`.

**3. One id, one row, and a narrated verdict IS the measured verdict.** Every
`were_you_right[].evidence_id` must name a `reads` item and its `verdict` must EQUAL that
row's measured verdict. Ids are derived from what they point at, so they can collide - a
store that appended a row twice, two congruence lines of one kind - and a collision is not
harmless: a reader keyed on `source_id` would silently take the LAST row, letting a narration
quote the second row's verdict while naming the first. So the pack MINTS ids through one
`_Minter` (the nth claimant gets `<base>#n`, nothing dropped and nothing raised) and the
night REFUSES a pack whose id names two rows. Every breach rejects the output WHOLE - not
trimmed, not partially kept: an unknown id, no ids at all, an evidence id no read row carries,
a verdict that disagrees, an observation graded as a call, a second claim on one read, a
`chased_against_news` other than `unknown` with no pasted forecast, an empty headline, a list
over its bound, any closed-schema breach. The last verified file stays byte-identical and the
ledger row is `degraded_no_narrative`. Two artifacts, two verdicts: a rejected day story never
costs the rolling D1 view its prior file.

**4. The reply is bounded HERE, and the bound comes from the evidence.** The shared
`ai_summary.validate_structured_output` enforces required keys, `additionalProperties`, types,
enums and a top-level string's `maxLength` - it does **not** enforce `maxItems`, nor the
length of an array's ITEMS. A 5,000-source, 500-claim reply was therefore accepted, written
whole (372 KB) and then drawn line by line on the Qt thread (reviewer round 1). TJ-4 is the
first consumer that renders a per-item list from a model answer, so it enforces the bounds
itself rather than widening the shared validator under every other caller. The working bounds
are the PACK's: at most one claim per read row and never two claims on one read, at most as
many sources as the pack carries ids, at most as many theses as the window carries D1 things
said - and the schema handed to the model carries those numbers. **A fixed cap could not be
right:** a regular session with every Mentor card answered already holds 8 read rows and 25
citable ids, because two of the six scheduled cards store TWO entries each, so the first fixed
pair (6 and 24) rejected a full day whole and left the trader with no story on exactly the
days they answered every prompt (reviewer round 2). `MAX_GRADED_CLAIMS` (64), `MAX_SOURCES`
(512) and `MAX_OPEN_THESES` (32) survive only as absolute ceilings on what a page can be asked
to draw. When a story grades fewer reads than the session held, the night counts it and the
page SAYS `graded K of N reads` - a size statement, with no result in it.

**5. No model runs by day, from any door - and "queued for tonight" is TRUE.** The slot
declares `uses_model` honestly, so `--force` cannot buy it the clock, and by day the page's
Redo button writes a `redo_requested` marker instead of starting anything. The first build
read that marker only for the session the night itself narrates, and the page's default pick
during a session day is the PREVIOUS session - so the DEFAULT click queued a day nothing would
ever read, for ever (reviewer round 1). Now the NIGHT SWEEPS: it scans
`sessions/*/redo_requested.json`, oldest first, re-narrates each (the marker overrides that
session's unchanged-hash skip), clears a marker only after a GOOD run, and names in its reason
what it narrated, what is still queued and how many wait for tomorrow. Three rules keep that
honest rather than merely busy:

- **The budget is spent on sessions NARRATED, never on names in a list.** Markers for three
  sessions with no pack took the whole budget and, being the oldest, sat at the head of the
  queue every night after - so the real request behind them was never attempted, on any night.
  That was live, not theoretical: the home folder held three session folders and zero packs
  (reviewer round 2). A packless marker now costs nothing, keeps its place and is named,
  `MAX_NAMED_UNBUILT` (5) of them and then `+N more`.
- **A Redo BUILDS the pack before it queues anything.** The night can only narrate a session it
  has a pack for, so the click runs the page's own off-Qt `build_pack_for` seam
  (deterministic, no model) and writes the marker - or starts the night's process - only once
  a pack exists. A build that produces none says `No pack could be built for <date> - nothing
  queued`. A host whose service has no `build_pack_for` seam still queues: there is nothing to
  build with, the click keeps today's behaviour, and a packless marker now costs the night no
  budget and is named.
- **A marker goes where the trader meant it or nowhere.** `validated_session` fails CLOSED:
  exactly `YYYY-MM-DD`, a real exchange session, already closed. `".."` used to write a marker
  at the `day_review` ROOT and `"2026-09-18-extra"` was truncated into a real day's folder. A
  folder that is not a session date is not a queue entry, and **no marker is ever retired by
  age** - uncertainty never deletes, so the reason line carries `+N more` for ever once
  markers accumulate, which is the honest cost of that rule.

A queued session's own failure keeps its marker and leaves that day's prior story
byte-identical; one bad queued session never costs the night's own story, its `ok` or the
rolling view; a night with no pack of its OWN still drains the queue. `--session` is the one
exception: an operator who named a day gets that day and no sweep.

**6. The window is asked again before every call after the first.** The slot's
`reserve_minutes` (10.0) buys the FIRST call and nothing more, so a night that narrated its own
session at 07:50 ET could declare five calls of nine minutes each and still be loading a model
at 08:35 - the night-only rule broken from inside (reviewer round 2). The order is own story →
rolling D1 view → sweep, and before the view and before each swept session the slot asks
`ai_jobs.window` for `SWEEP_CALL_MINUTES` (9.0, the per-call timeout) of room. Measured against
the real off-hours window: the first call is allowed at 07:50 ET and refused at 07:51, every
later call allowed at 07:51 and refused at 07:52, so with `TIMEOUT_SECONDS` = 540 the latest a
call can END is 08:00 ET, exactly the close. A window that has closed stops the run CLEANLY:
every remaining marker is kept and the reason says the window closed and what is still queued.
The clock is ONE injectable seam - the slot's `now` plus the run's own monotonic elapsed - and
nothing sleeps.

**Two size rules, both stated rather than hidden.** The pack carries the three walk-away rows
that ran furthest after the decision (an ORDER, never a ranking that decides anything), and the
rolling D1 view sees only the D1 items of the last `LATELY_SESSIONS` exchange sessions - never
an M5 item, because a rest-of-day read is about the tape and not about the bigger picture.

**Where the slot sits is a measured choice, not an append.** `plan.md` said "after
`market_story_narration`", but gate #158 wants the day story finished before 23:30 Pacific and
`ticker_briefs` reserves 120 minutes in front of it. It cannot move further forward either -
two existing pins say `ai_summary` sits directly after `measured_report` - so it goes between
them, INSIDE decision 0018's stage 2, after `ai_summary` and before `observation_tags` and
`ticker_briefs`.

### The four rounds, in plain words

Every round was NO-GO, and every round was the reviewer finding the thing that would have
happened on a real night rather than a thing that reads badly.

**Round 1 - the button lied.** By day the Redo click said "queued for tonight", and for every
session except the one the night was about to narrate anyway, nothing would ever read the
marker. The page's default pick during a session day is the PREVIOUS session, so the DEFAULT
click was the broken one. Round 1 also found the unbounded reply: a schema that validates keys
and types but not list LENGTHS let a 372 KB answer through to the Qt thread.

**Round 2 - the fix for round 1 starved itself, and the caps were too small for a good day.**
Three markers for sessions with no pack ate the whole sweep budget and, being oldest, sat at
the head of the queue for ever. That was not a thought experiment: the live home folder held
three session folders and zero packs. In the same round the caps chosen in round 1 (6 claims,
24 sources) turned out to be below a REAL full day - 8 reads and 25 ids - so the trader would
have lost their story on precisely the days they answered every prompt. And a worst-case night
declared five nine-minute calls against a ten-minute reserve, which is the night-only rule
broken from inside the job that declares it.

**Round 3 - the trader's own hand.** `redo_story()` built a NEW `_RedoPackWorker` on every
call and nothing tested for one in flight, so three impatient clicks started three full
`read_day` builds racing on one `pack.json` and three `run_ai_jobs.py` children, with the
button enabled throughout. Now `_redo_busy` makes a second click a NO-OP that leaves the
"Building …" note exactly as it is, the button is grey from the click, and `_release_redo` runs
on EVERY ending - queued, launched, no pack, refused, a launcher that raised, a build that
raised - through a `finally` plus a backstop in `_drop_redo_worker`, so a worker that ends
without its `done` reaching the page cannot leave the verb grey for ever. A session switch
mid-build neither frees the button early nor strands it. The same round's advisory became a
rule: `validated_session` now gates the LAUNCH branch too, and `run_ai_jobs.py --session` is
that same function, so the picker's provisional Today is refused at the door instead of
parsing and answering `skipped` later.

**Round 4 - the product was whole and the TEST was the blocker.**
`test_a_session_switch_mid_build_neither_frees_nor_strands_the_button` counted
`service.calls` while the worker thread was still on its way to its append: 2 failures in 8
runs, and the one test covering the lead's session-switch requirement could not be trusted. The
lead fixed it at `6c1e2506` - the gated service sets a second Event after its append and the
test waits on it - and measured the fail-before-fix the previous commit had promised and never
recorded: 9 failed / 5 passed on the reverted code, 14 passed on the fix, 25 of 25 green in the
round's own run.

### Two live facts this lands on

Measured on the desk, 2026-09-20: `C:\TradingBotData\day_review\sessions\` holds **three
session folders and ZERO `pack.json`**, so the first Redo after this merge is also the first
pack build for that session, and gate #148's third clause is readable for the first time. And
**no live journal row yet carries a TJ-14A prediction**, so the first morning's were-you-right
list will be EMPTY unless the trader clicks a prediction on the Mentor card first. That is the
honest empty state, not a defect: the story still has a headline and the deterministic facts,
and `graded 0 of 0 reads` is a true sentence about a day nobody made a call on.

### Recorded, not repaired

- `build_pack_for` runs a second full `read_day` for a day the panel has already read - off
  the Qt thread, but duplicated work.
- `redo_story()` reads the settings and the clock and writes the marker on the Qt thread, on
  click.
- A build that RAISES is reported with the NO-PACK note; the exception survives only in the
  log line. The note is true (nothing was queued, no pack exists) and the button comes back.
- `_sweep_queued` iterates every packless marker each night, one `read_pack` per entry. Cheap
  and bounded by the folder count, and each one is named - but the queue is never retired, by
  design.
- `run_ai_jobs.py --session` now fails CLOSED on an unanswerable calendar (`2099-01-01` → exit
  2), reversing the deliberate fail-open comment the round-1 code carried. Intended by the
  lead's "one shared rule" decision, and the validated NYSE range is 2000-2032.

## TJ-12 - the report card, and why every line quotes somebody else (2026-09-20, packet TJ-12)

The trader, 2026-09-19: *"I want what I missed to be very apparent. I want what I did well
with to also be very apparent."* The card is the answer, and its whole design rule is that
**it measures nothing**. Six lines head Day Review above the story - Did well, Missed, Your
reads, Congruence, Process, How fresh - and each one is a quotation with a count beside it.

**Why no new statistic.** Every number on this card already exists somewhere on the desk with
a rule behind it. A card that re-derived a run rate would be a second opinion about
`REAL_MISS_V1`; one that recomputed a Wilson would be a second interval over the same numbers
and the first thing to drift the day `swing_headline`'s z moves. So `did_well` prints
`walkaway_day`'s own `liked_not_traded` sentence and its skill sentence, and the family it
names carries that CELL's own `low`. The tester's fixture is deliberately a trap: `steady` is
55/100 (bound 0.4524) and `flashy` is 18/30 (rate 0.60, bound 0.4232), so a card ranked on the
RATE names the wrong family, and `tiny` is 9/10 - the best rate on the board and under the
floor, so a card that ignored `MIN_REPORTABLE_N` names it. Neither happens.

**Why the DAY line carries no `rate_lb` and the WEEK line does.** A Wilson over one session is
a statistic the card invented. A week POOLS: the tester's two sessions are four clicks and six
on purpose, because that is the only shape where pooling (4/10 = 0.400) and averaging (0.458)
disagree. `week()` sums the counts, de-duplicates the session, and computes the ONE Wilson
from the pooled pair.

**A ledger status is a vocabulary this module does not own.** The first build tested
`status != "ok"` and named **22 slots broken for 2026-09-18; 2 were**. Twenty of them had
finished `ok` hours earlier and were `skipped` by the next half-hourly pass - `daily_digest`
ok at 22:02:10 and skipped at 03:30:41, `weekly_synthesis` ok at 02:52:23 and skipped three
minutes later. Only `journal_import` (failed three times) and `ai_summary` (degraded) were
really not ok. `ai_jobs/ledger.py` owns this vocabulary: `STATUS_OK` is the only completion,
`ATTEMPT_STATUSES` is the owner's own "something went wrong" set, and a skip is what the
runner writes whenever the window or the already-done check says there is nothing to do. So
the card reads the owner's constants and spells none of them, names `failed` and `degraded`
SEPARATELY (a degraded run published a real document with no narrative - calling that "nothing
ran" is a different fact), and lets the LAST deciding row win, so a slot that failed and
recovered is fine and one that ran and then failed is not. **The line whose job is to say when
the night failed is the one line that must never cry wolf.**

**A night the window cannot reach is `unknown`, never quiet.** The second build set
`night_status` on `target.exists()` alone, so a session whose rows fell outside the tail read
*"0 overnight slot(s) read, 0 finished ok, none reported trouble"*. On the live ledger
(1,256,082 bytes, 483 rows) the 256 KB / 500-row window holds **173 rows and reaches back only
to 2026-09-11**, so **9 of the 15 sessions the Day Review picker offers** reported a clean
night over 10-16 real slots each, and 2026-09-11 was half-visible and said 5 finished ok when
16 did. Same defect class as the status bug, in the same line. `_tail_rows` now returns a
`_LedgerTail` that says whether the read TRUNCATED and the oldest `session_date` it saw, and
`how_fresh` has three answers: **unknown** past that edge (the asked session `<=` the oldest,
because the window may have cut that night in half - exactly the 2026-09-11 case), with no
failed slots and **no slot counts at all**, since a count there would measure the window and
not the night; **`no_rows`** for a covered session that is genuinely empty; and "none reported
trouble" only ever over at least one slot actually read. There is no second read and no
whole-file fallback - the file is opened exactly once either way - and truncation is a fact
about the READ, never about the date, so a small ledger read whole still gives an old session
its real counts.

**An unread lane is not an answer.** `trade_origin.planned_state` says `unplanned` whenever
nothing precedes the first fill - and the desk can only read two of the four lanes, so **30 of
the trader's 33 trades since 2026-08-20 came back `unplanned`** (17 sessions, mean 1.8 a
session) and the woken Mentor was about to ask about every one of them. Missing data read as
confirmation is exactly what plan.md sec 5 forbids, and the fix is not to guess: the card says
what it DID look at, names what it did not (`DESK_ORIGIN_LANES_READ`, one constant both lane
builders build from), carries `lanes_read` / `lanes_unread` so TJ-5 and the pack say the same,
and the question leads with the same caveat and offers `a_focus_pick`. When **TJ-12F** fills
the two lanes the plain wording comes back on its own - no re-wording, because the sentence is
chosen from what the caller DECLARED it read.

**Why `How fresh` exists and why it reads a tail.** The trader's second look added it: a page
that shows yesterday's numbers without saying the night failed is a page that lies quietly. It
names the story's stamp, the last session the desk has VERIFIED fill coverage for
(`trade_mentor_trade_check.fills_current_to` - an absence is NOT a date, so it says "no
verified coverage yet" rather than printing today), the session the reads were graded through,
and any slot whose LAST deciding row for the session is not `ok`. `ai_jobs.ledger.recent_rows`
reads the whole file before slicing it, which is right for the runner and wrong for one
sentence on a page, so past `LEDGER_TAIL_BYTES` (256 KB) `day_report_card` seeks the tail
itself and `ai_jobs/ledger.py` is left unchanged for its other callers. And a missing store is
`unknown`, never "fine": `ledger_path()` defaults to `create=True`, so the reader whose honest
answer may be "I cannot see last night" must not be the thing that creates the folder it is
asking about.

**Why it is on the worker.** TJ-1's rule: Day Review reads ONE payload on ONE worker. The card
is another projection of that payload, not a second read. The two file reads it needs - the
read ledger through `prediction_ledger.your_reads` (so the tally travels as integers and the
page counts nothing) and the job ledger's tail - belong to `read_day` and to nowhere else. The
page's `render` formats six strings. Every line is guarded on its own, so one owner raising
costs its own sentence (`could not be read: ...`, `measured_ok: False`) and a failed card
costs the card, never the day.

**Why the wake happened here.** `mentor_questions` registers a kind with `dormant_until`
naming the packet that builds its reader, because a question whose answer nothing reads is the
trader's time spent for nothing (decision 0021 answer 28). TJ-12 IS that packet for
`trade_origin` and `open_position_check`, so the field is cleared and the lane is filled.
Filling the lane is not optional: `planned_state` answers `unplanned` when nothing was said,
and an unread store looks exactly like nothing said - waking the kind with an empty lane would
have asked the trader where EVERY trade came from.

**What changes for the trader on MONDAY, once this is live.** The card heads Day Review, so
the first thing on the page is what they did well and what they missed, each with its `n` and
each clickable to the table it came from, and `How fresh` says underneath whether last night's
work actually ran. On the Mentor card (DESK / EVENING / OFF, never AWAY) two new questions
become possible: *where did this trade come from* about a trade the desk cannot connect to any
claim or like before its first fill - about **two a session** on the last 17 sessions'
evidence, comfortably inside the budget of three, asked once per trade, with `Stop asking
this` always available and the prompt SAYING that Focus adds and armed alerts are not read
yet - and *is the thesis intact* once a week about a position still open past five exchange
sessions. An unanswered long hold reports as `unanswered`, never as "the thesis is intact",
which is a claim nobody made. And the Process line no longer hands the trader a daily
scoreboard calling them impulsive: it says *"2 planned (a claim or like before the first
fill), 1 with no claim or like before the fill"* and then names the lanes it could not read.

## TJ-5 - the week is five days, and a day nobody packed is not a quiet day (2026-09-20, packet TJ-5)

The trader asked for *"5 of these days collated into one tab … to see if I was right, to see
if I chased in bad news environments, and to compare what I actually said to what I did"*
(2026-09-17). Weekend Prep's first step is that tab. The shape is TJ-1's: ONE payload, ONE
worker, one render and no special cases - `reload()` on this page used to BE the read, and
was the worst measured stall on the desk (8.45 s of frozen GUI, fluidity capture
2026-08-25).

**Five cards, always.** A card is a DAY, not a pack. The live day-review folder held three
session folders and ZERO `pack.json` on 2026-09-20, so the first thing the trader sees is
`0 of 5 sessions have facts` - and that has to read as honest rather than broken. A day with
no pack says `not packed` and its tally says `no reads graded`; it is never blank (which
reads as a quiet day) and never zeroed (`0 right of 0` is a measurement nobody made). A day
that does not EXIST - the fifth weekday of a holiday week - shows its heading alone rather
than the double negative *"no session this exchange week: not packed"*.

**The page computes nothing.** Every number on it was computed by the module that owns it.
The week-level numbers come from `week_review_narration.build_week_inputs` - the SAME
function the Saturday slot narrates from - so the page and the story cannot disagree about
how many days had facts. The strip's pooling is `day_report_card.week_from_cards`. The
tests prove the boundary by monkeypatching `day_review_pack.build_pack`,
`day_report_card.build`, `day_report_card.week` and `run_week_review_narration` to RAISE;
the page still renders. The day chart's bars come from TJ-2's durable parquet only
(`day_review_bars.read_session_bars`): a page that reached a provider on open would be five
network calls behind a tab click.

**Why `week_from_cards` exists.** `day_report_card.week(sessions)` needs TJ-11's
`WalkawayDay` OBJECTS: `_rows_of` is `getattr(walkaway, name, ())` and `_skill_window` reads
`walkaway.skill`. A day pack carries neither - it carries the BUILT lines with all of their
integers. So the week cannot be re-cut by re-reading twenty packs and calling `week()`. The
pooling therefore lives once, in the module that owns what a line IS, and `week()` delegates
to it. The half it cannot do is the best FAMILY, which needs the walk-away skill cells: a
week pooled from cards names no best family and SAYS why
(`day_report_card.NO_FAMILY_FROM_CARDS`), because a best of five day-winners is a ranking of
days, not of families.

**An unreadable day is not a quiet day.** `day_report_card._unreadable_line` marks a line
the desk could not build (`measured_ok: False`) so the page says which of the six sentences
failed rather than showing six placeholders. Pooled into a week, that marker used to be
DROPPED and the cell read `n 0 / measured 0` - which is exactly what a day the trader did
nothing on looks like. `_pool_cards` now carries `unreadable_sessions` on every pooled line
(present and empty when there are none) and says `K day(s) could not be read` when there are
any. The COUNTS do not move: a day nobody could read measured nothing, and adding it to `n`
would invent a denominator. `week()` and `week_from_cards` share `_pool_cards`, so TJ-12's
own week path gained the marker in the same edit with every TJ-12 test untouched and green.

**`how_fresh` is asked per session, by the strip.** It is deliberately not in the day pack:
it describes the machine's night and its text moves every time the job ledger gains a row,
which inside the pack's hashed body would move `inputs_hash` and buy a model call to
re-narrate the same session night after night. `_slot_verdicts` guards with
`if session and …`, so an EMPTY session pools every night the ledger tail holds - a five-day
window built that way would multiply its own `n` by five (the advisory TJ-12's third review
round recorded for this packet). The strip therefore builds each day's card from the pack's
five stored lines and asks `how_fresh` itself, with that session, through a ledger opened
`create=False`: a reader whose honest answer may be *"night status unknown"* may not make a
store in order to say so.

### The week story: one call, and a schema is still not a guard

`week_review_narration` is TJ-4's pattern at week scale. Its inputs are the five packs, the
five day stories, the weekly rollup and TJ-15's/TJ-16's contrast packs; `EVIDENCE_KEYS` is a
closed tuple and a test asserts that no bar or lake name can be in it. Each day travels
BOUNDED - at most `MAX_ITEMS_PER_DAY` said items and read rows - and what did not fit is
COUNTED and said in the day's `omitted` block, never silently dropped; a day's `tally` is
counted over its WHOLE read list, never the bounded copy. Every citation is
SESSION-QUALIFIED (`week_source_id`, separator `/`): each pack mints ids with its own
minter, so `said:<entry>:prediction` repeats across the five and an unqualified citation in
a week story would name two rows. The bounds handed to the provider come from the input
(`_schema_for`) and are re-checked afterwards in `check_week_narration` - `maxItems` and
`additionalProperties` are a grammar hint to the decoder, never a guard, and three packets
shipped that bug on 2026-09-20 alone. A rejection costs the NIGHT and never the file the
trader read last Saturday.

The reviewer attacked the narration **21 times and was rejected 21 times** - a fourth
tendency when three were offered, a tendency `n` bent by one, an invented and a duplicated
tendency cell, `were_you_right.right` plus one, an empty headline, empty `sources`, an id
from a session outside the week, the SAME id unqualified, extra keys at depths 1, 2 and 3,
10,000-item arrays and 9,000-character strings - with the prior week file byte-identical
after every one, no `.tmp` left behind and the status always `ledger.STATUS_FAILED`. A
replay of LAST week's own valid reply is rejected on its first out-of-week id. Below the
floor (`MIN_NARRATED_DAYS` = 3) the slot writes a deterministic scaffold holding
`narrated K of 5` with a falsy `narration` and status `skipped`, and loads no model at all -
measured at **0 model calls**; a run that does narrate makes exactly **one**, and a second
run on an unchanged hash makes **none**.

**A reserve is not a timeout.** The unmeasured default was 30 minutes beside a 1,800 s
timeout - the same thirty minutes - so a call that ran to its own timeout consumed the whole
reserve and left nothing for the model load, the validation and the write.
`DEFAULT_RESERVE_MINUTES` is derived from `TIMEOUT_SECONDS` plus `RESERVE_MARGIN_MINUTES`
(40 min), so the two cannot drift, and a TJ-13B probe measurement still overrides both.

**Which model.** `provider.week_review_plan()` answers `may_run_large: False` until TJ-13B's
probe row exists. That costs the LARGE model and never the story (lead decision, 2026-09-19:
the trader wants a week story every Saturday), so the slot asks the MEDIUM tier and the
plan's reason is written to the row under `provider.LEDGER_FIELD`. `openai` is refused by
the provider seam before anything is sent, and the slot records a FAILED row carrying that
sentence rather than crashing the night.

**Cost, measured and not repaired.** On a desk-sized scratch tree (20 sessions of packs,
191,930 bytes) the whole payload took **0.029 s** with a traced peak of **0.5 MB** and came
to 11,904 bytes of JSON - but it made **37 `read_pack` calls per page open** for at most 20
unique sessions, because the four week windows, the month-to-date and `build_week_inputs`
each re-read. That is RECORDED here rather than repaired: it is 29 ms today and it scales
with real pack size, so it is the first thing to look at if this page ever feels slow.

**The lesson from integration.** A targeted run cannot see a source-count pin in another
packet's test. The sparkline set a minimum height of its own; R4's one-owner rule for the
table floor is pinned by counting the SETTERS
(`test_the_ten_row_floor_is_one_constant`), so a second setter in a new widget broke a test
in a file TJ-5 never touched and no `-k tj5` run could ever have shown it. The lead's full
suite on the trial merge caught it and the fix (`e93ffe49`) states the floor as a size HINT
instead. When a packet adds a widget with a size rule, the full suite is the only proof.

**What the trader reads first, on a desk with zero packs.** Today's live
`C:\TradingBotData\day_review\sessions\` holds three session folders and no `pack.json` at
all, so the page opens on `0 of 5 sessions have facts`; five cards each reading
`<date>: not packed - the desk has no facts for this day.` with `no reads graded`, chased
`unmeasured`, said vs did `unmeasured` and an empty chart; the week-story panel reading
*"No week story yet. The Saturday night slot writes it from the week's own packs; nothing is
fetched or narrated by this page."* (after the first Saturday night it reads *"No week story
yet - narrated 0 of 5. The night writes one once three of the week's days have a story of
their own; below that the desk says the count rather than narrating days nobody
measured."*);
a week/month strip whose every cell says `n 0 - measured 0` with `how_fresh` reading the
night status it can actually see; and a summary saying `WALK-AWAY THIS WEEK: no session
packed yet.` with the unpacked days named. Nothing on that page claims a measurement. It
reads as a desk waiting for its first packs, which is what it is.

## TJ-3 - a mark sits on a bar only when it happened DURING that bar (2026-09-19, packet TJ-3)

The long form behind the CLAUDE.md rule *"A note marker sits on a bar only when it
happened DURING that bar."* Trader, 2026-09-17: *"it also needs some sort of chart system
to show me when I commented on it so I can see exactly where I went wrong."* plan.md §12.4
TJ-3, decision 0021. Branch `claude/tj3-note-markers`, tip `58ee11f4`, merged `72647104`
into `lead/p033-integration2`. Two reviews, both GO by reproduction, the second after a fix
round.

**One rule decides everything else here: a marker's `index` is an index into the tape it
will be DRAWN on, and it exists only when the stamp fell INSIDE that bar's own span.** A
name whose session tape is short does not borrow SPY's index - that index names a different
moment and would fall past the end of the drawn bars - and the stamp is not CLAMPED onto
the last candle either. The tester's first literal said bar 44 (SPY's index for that
stamp), the lead amended it to 29 (AAA's own 30-bar tape), and the review round showed the
honest answer is that the stamp belongs on NO bar at all. What the clamp cost was measured
on a copy of the live journal: 62 of 216 trade legs (29%) fill after 13:00 Pacific, and
every one was being drawn on the 12:55 candle as though the trader had acted at the close.
The same probe found `trade_date` is the EXIT day and 116 of 216 `opened_at` values fall on
an EARLIER day - those correctly place nothing on the session's own tape.

So `placement_for(bars, stamp)` answers `(index, placement)`: `on_bar` (drawn),
`after_tape` and `between_bars` (KEPT with `index: None`, counted, drawn nowhere), and
`before_tape` / `no_tape` / `unreadable` (no marker at all - a pre-open thought says nothing
about a candle and there is nothing to count it against). `bar_index_for` delegates and
keeps that meaning. The intraday test is `start <= stamp < start + width` with the bar WIDTH
read off the tape itself - the smallest positive gap between starts, so a halt does not
widen a bar - and a DAILY tape matches by market-local DATE. Comparison is always
`astimezone` with market-local ATTACHED to a naive stamp, never a stripped zone: the journal
stores UTC and `day_review_bars` persists `America/Los_Angeles`, so 17:12Z is the 10:10 PT
bar, while a naive read made it bar 77, the last of the day. On a DAILY tape a weekend or
holiday stamp is `between_bars` rather than Friday's candle - the page has no D1 toggle and
this packet added none, so that is a decision owed KNOWINGLY if one ever lands, against the
trader's TJ-11F rule that a Friday-evening call is Friday's judgement.

**What could not be drawn is SAID**, because a silent drop is the same lie as a clamp with
fewer pixels. `placement_counts` counts the placements ON THE WORKER, `name_charts` carries
them per name and `spy_marker_placements` carries the benchmark's;
`day_review_panel._marker_caption` turns them into `2 marks after the tape - not drawn.`
(joined with `N off a drawn bar` when both happen), and a session with no tape at all says
so instead of nothing. On the live 2026-09-17 session: 278 marks over 174 name charts =
270 drawn + 8 counted.

**`ref_id` selects, `marker_id` addresses.** Both legs of a trade carry the same `trade_id`
- that is the id the page selects with, and a leg click deliberately selects nothing,
because a trade is not a note - so each glyph also carries `<trade_id>:in` / `:out`, which
`note_marker_position` matches first. Without it the entry leg answers for both and the exit
glyph can never be found.

**The kinds are a vocabulary, not a judgement.** Ten come from `plan.md`; the eleventh is
`prediction`, because a Mentor read the trader DESCRIBED and one where they clicked a
direction are not the same thing (TJ-14 item 1). A Mentor marker is placed at
`responded_at`, falling back to `created_at`, and NEVER at `scheduled_at` - an unanswered
prompt is not a thing the trader said. `benchmark_markers` builds the SPY chart: market
notes, Mentor reads, the pasted forecast and EVERY trade of the day, each labelled with its
symbol. `symbol_markers` builds one name's own decisions, claims and trades, and
`name_charts` pairs each acted-on name with the tape that holds it, so a name the bars file
never got is absent rather than drawn on somebody else's tape. `dislike` and `not_today`
draw in the rejection family beside `veto`, `swing_favorite` in the like family (lead,
2026-09-19), with the verdict still spelled out in the label. A machine row never gets a
marker. `capture_id` is present-and-empty on every pick-feedback and swing-favorite row, so
the id falls back to what the row IS - a marker two decisions share is a marker the page
cannot select with.

**`candle_chart.py` is the Alert Center's chart too, so this is additive or it is
nothing.** The reviewer proved it by reproduction at base and at tip: for an unmarked D1
chart (earnings, levels, overlay, volume) the scene items, the view range, the earnings
glyph positions, the ribbon span and the sha256 of the full-widget PNG are IDENTICAL, as are
the bar / level / empty-space click sequences; with no markers set nothing is built and
`_position_note_glyphs` and `NoteMarkers.paint` run 0 times. Once a payload arrives the
glyphs pool like `_sync_earnings` (hidden, never destroyed), every item is added
`ignoreBounds` so no marker moves the y-range, and `set_data` DROPS the payload because new
bars mean every index in it names a different moment - the host pushes markers after bars.
`markerClicked` is emitted IN ADDITION to `barClicked`, `priceClicked` and `levelSelected`.
The one shared line that changed: `EarningsDropLines`' ribbon fraction moved from a module
constant to a class attribute of the same value, so the note connectors take their own line
(0.20 against the earnings 0.10). `tests/test_tj3_chart_is_additive.py` is the proof.

**Everything is built on the worker.** `read_day` resolves both payloads at the END of the
one read, against the tape that read ENDED UP with, from rows it had already opened - no
second store and no read on a row click; the paint path never imports `day_review_markers`.
The page formats: ONE reused `_name_chart` beside the tables, and a marker click selects
that note in "What you said" while a marker for something that is not a note moves no
selection. Measured cost: `read_day` median ~502 -> ~554 ms on a staged session, and a
payload of ~1.0-1.5 MB for a 174-name session because `name_charts` carries a tape per
decided name. **Recorded as a LATER packet:** trimming `name_charts` to the rows the tables
actually show.

Live gate **#147** is owed and needs gate #146/#152's back-fill first: the live home has no
`day_review/bars/*.parquet` yet, so with no file the SPY chart draws from the Qt hand-off,
the caption says the marks were not drawn, and no name chart can open.
