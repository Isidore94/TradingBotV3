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
- **The overnight runner's `veto_cohort_grading` slot is deterministic and calls no model.** **The order is decision 0018's: deterministic slots, then the digest, then narration, then the model-gated slots; a later phase appends inside its stage and never reorders across stages** (`docs/decisions/0018-deterministic-stage-before-narration.md`, 2026-09-04 - the two narration slots held up to 2½ h of reserve ahead of every deterministic slot, a slot that cannot fit its reserve records SKIPPED, the 2026-09-01 run took six hours, and no deterministic slot reads either narration slot's OUTPUT). The order is pinned once, as `EXPECTED_SLOT_ORDER` in `tests/test_ai_jobs_runner.py`. Nothing in this chain may reach a detector, score, alert, watchlist, Focus, the review queue or `review_policy.json`.
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

### The three auto-tagging lanes, long form

- Auto-tagging has three lanes that never compete and are ordered by LANE, never confidence: `journal_analytics.AutoTagger` (which setup, from the scanner's own files), `journal_trade_shape` (facts from the trade's own timestamps and legs), and `trader_capture` (what the trader already SAID inside the trade's own window, outranking every fuzzy source, a rejection prefixed `vetoed:` / `passed:`). No tag is ever derived from the outcome; unmeasurable emits NO tag; `context_row_id` is a pointer, and plan.md P5.3/P5.4 own the canonical opportunity id. `preference_trade_outcomes` shows its match confidence on every row or says "no match".

### A broker file is authoritative for money, long form

- **A broker file is authoritative for money and blind to time.** `journal_statement_import` (Questrade `.xlsx`, no `openpyxl`) and `journal_ib_transactions` (IBKR sectioned csv, USD price / CAD money, masked accounts unmasked only when exactly one fits) write executions at MIDNIGHT market-local; `journal_trade_shape.is_date_only` refuses to name a session for them. Side and options come from the DESCRIPTION; identity is `fill_signature` plus an ordinal, never positional; a statement never writes into a (broker, account, day) a richer source already covers. Commission carries a SIGN and the importer owns it — nothing downstream may `abs()` it.

### M1 - the challenger measured through the catch-up path, long form

- **The AVWAP band challenger is measured through the CATCH-UP path too** (M1, 2026-09-05): `build_anchor_band_variant_meta` lives in `legacy.py` and serves BOTH the live scan (`runner.py` re-exports it) and the tracker catch-up (`_evaluate_priority_snapshot_for_date`), which never set the block, so the shadow measured nothing from 08-26 to 09-05. Never add a third builder; a record is rebuilt on every persisted tracker write, so no migration exists. The Band variant tab prints `Measured N of M setups (K unmeasured: <top reason>).` from the export's own counts (`top_unmeasured_reason`), never by reading the 1.1 GB tracker; still shadow only, T4's 20-session accrual starts at the first measured row.

### M5 - the control, study and exit-framework populations, long form

- **The control, study and experimental-exit populations are SURFACED, LABELLED, and never mixed with picks** (M5, 2026-09-05): three Setup Tracker tabs - Controls (`N graded episodes from the M setups`, never one number under the other noun), Studies, Exit frameworks (`comparison_apr2026` beside `baseline`, `n_filtered_by_experiment` reconciling the two n's) - read three CSVs written in the tracker's own guarded save pass. Win rate leads with `n` and the ONE Wilson bound, the sort is the bound, each tab carries a population sentence and `experimental` is a COLUMN. Shadow only; the champion aggregates are pinned byte-identical.

### The M5 Strength Board's auto-adoption, long form

- **M5 Strength Board:** batched yfinance over `universe_all.txt` PLUS the four trader watchlists, zero IB traffic; relative volume is SESSION-RELATIVE and is not one of the seven fenced formula functions (byte-identical to the R8 baseline); D1 SMA floors read `2y` with today's forming bar dropped. Its parity rows auto-join M5 Focus (`_auto_adopt_strength_board`: DESK only, empty `failed_floors` only, the ONE adoption gate re-run per row, skipping `_ignored_symbols` and `FocusPickStore.declined_today`, one `add_many` per side plus `mark_auto_adopted`, never `FocusService.add`, never removing). Every Focus add is injected into `longs.txt` / `shorts.txt` by `FocusPickStore._inject_into_shared` and a removal un-injects it.

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

## SN - the scanner's hold on the interpreter (2026-09-08, SN5/SN6 built; SN1-SN4 open)

- **What the trader said:** the desk was *"really quite laggy"* at the close; then, on 9% of the week's usage, *"Anything we can get done from wishlist.md that's quick and cheap?"* and *"Go"* for SN5 and SN6.
- **What was measured (2026-09-08, `thread_cpu.jsonl` / `ui_stalls.jsonl`):** `Thread-4 (run_strategy)` at 0.62 of a core on average in hour 13, 71-88% per minute at the close, climbing from 0.35 at 06:00; the GUI thread at 0.10-0.15; 13,031 GUI stalls over 50 ms (four hours hit the 2,000-record cap, so undercounted), 245-365 blocked seconds per hour, one 30.6 s stall at 13:01:53. Cycle 24's preamble was 658 s: the fast lane 333 s over 258-259 names (107 auto-adopted + 64 + 129 trader files, alphabetical), then four RRS passes 275 s. A cycle is ~25 min and `wait_for_candle_close` returns at once, so the loop never rests.
- **Why a thread cannot be made polite:** the scanner shares the interpreter lock with the GUI; a CPU-bound thread releases it only at the interpreter's switch interval, and no priority trick frees it (the F1 lesson). The full fix is SN1, the scanner in a below-normal child process. Until then the thread must GIVE the lock away.
- **SN5 - the breath.** `BounceBot._breathe` waits `SYMBOL_BREATH_SECONDS` (0.02) on `_stop_event` after each symbol's compute, in the fast lane and in both main-sweep loops. On the stop event and never `time.sleep`, so a set event returns immediately and shutdown is not one symbol slower. Cost: ~586 symbol scans a cycle x 20 ms = ~12 s on a 25-minute cycle. Pacing only: the loop, the set, the bars and every output are unchanged; `ScanCycleClock` still never sleeps.
- **SN6 - the trader first.** `BounceBot._fast_lane_order` returns the trader's own Focus names (no marker in `focus_auto_picks.json`) alphabetically, then the auto-adopted ones alphabetically; the sweep follows as before. The engine reads the markers through `focus_picks.load_auto_pick_symbols()`, which shares `_read_todays_auto_pick_markers` with the store, so both apply the same per-entry `session_date` rule (R2.1). A failed or missing read is an EMPTY set: every name then scans as the trader's, which promotes and never demotes or drops a name - absence of a marker means the trader owns it (R2). The fast-lane log line prints both counts.
- **What did not change:** `request_and_detect_bounce`, the RRS passes, `_rebuild_feed`, every detector, threshold, tier, fold and evidence row. SN1-SN4 remain in WISHLIST as ideas until the trader moves them into `plan.md`.
- **File-scoped ask-first:** `bounce_bot_lib/legacy.py` is a detector file; the "Go" of 2026-09-08 is the yes for these two seams only.
- **Tests:** `tests/test_sn5_sn6_scanner_breath_and_fast_lane_order.py` - 12 of 13 fail with the fix reverted (proved on the lead's checkout, 2026-09-08).

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
