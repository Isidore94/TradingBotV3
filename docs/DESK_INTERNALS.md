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
- **The nightly journal slot speaks, and the Questrade chain is watched** (AI-layer review packets, 2026-08-24 — `docs/analysis/AI_LAYER_REVIEW_2026-08-24.md`): `run_journal_backfill` returns `failures` beside `status` and the nightly ledger reason names the first three failures and PRINTS the count it dropped, plus only what the night measured — a skipped reconcile says "skipped", never "0 mismatches". Reconcile MISMATCHES never marked the slot failed (refuted premise, regression-pinned); only exceptions do. `scripts/journal_health.py` surfaces a dead Questrade OAuth refresh chain in Journal ▸ Health and System Health; repair is a TRADER action — paste a fresh refresh token into Journal ▸ Health ▸ "Questrade refresh token" (key `journal_questrade_refresh_token` in `%LOCALAPPDATA%\TradingBotV3\local_settings.json`, a secret-bearing file; single-use rotating chain, weekly paste is the routine, never the env var). **The chain has ONE owner** (2026-08-25): Questrade rotates on every refresh — a success invalidates the access token it replaces AND consumes the refresh token it was given — so "Pull today now", the gap backfill and the nightly slot were three consumers of one single-use chain and broke it eleven minutes after a paste (import OK 20:54:59, backfill 20:59, `400` on the refresh endpoint 21:06:51). `refresh_access_token` now holds `local_writer_lock`, **re-reads the token inside the lock** (a caller that waited spends what the winner LEFT), and writes the four rotated values through `project_paths.save_local_settings()` in one temp-file+`os.replace` save; `_authorized_get` answers a 401 caused by someone else's rotation by picking up THEIR access token instead of burning a refresh. A failed refresh saves nothing and never clears the stored token. **The attempt cap counts failures against a DAY, not a cause**, so days that failed while the chain was dead were skipped forever — `self_heal(include_exhausted=True)`, passed only by "Retry failed Questrade days", lifts it for one deliberate run while the nightly keeps it; `attempts` is never rewritten and the run reports `reopened_exhausted`. **Not every FAILED day is repairable:** 44 of 45 `activities report trades…` days predate 2026-06-10, the executions endpoint's retention horizon, so retrying them can never work — recovering them from `/activities` or labelling them permanently uncovered is a trader decision needing a new coverage status. Lesson paid for twice: address home-folder stores by their `project_paths` named constants — Focus Pick Review resolved CSVs by name under `PERSISTENT_DATA_DIR` while they live in `data/runtime` and shipped a blank page from 08-18 to 08-24, and its fixture encoded the same wrong assumption.
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
- **A sweep-finalized trade counts under the policy that MEASURED it** (trader Decision A, 2026-08-25 — `docs/analysis/POST_ATTACK_AUTHORIZATION_2026-08-25.md`). The after-close sweep finalizes with a BLANK eod-hold `close_r` by design (`no_eod_close`): with no bars through the close there is no such number, and inventing one would make the same trade report differently depending only on what the finalizer held. What it did measure is in `context.exit`, not `context.path.exit_policies`. `setup_scoreboard.exit_policy_r` derives three frozen policies per final — `eod_hold` (the settled `close_r`), `stop_exit` (the STORED `context.exit.stop_exit_r`, read only where `exit.stop_hit`), `last_measured` (`(last_measured_close - entry_price) / risk_per_share`, sign flipped for a short) — as columns `r_eod_hold`/`r_stop_exit`/`r_last_measured`. **`usable` = at least one policy measured the row**, still ANDed with the risk floor and the R10.B claim split; unresolved rows stay unusable and are counted by reason. **They are never blended**: one table per policy through `evidence_stats`, and every eod-hold ranking view reads `r_eod_hold`, never `close_r`, so a row with no EOD close cannot widen an eod-hold n. Keying `usable` on `close_r` made all 656 of the 2026-08-25 finals invisible to every evidence surface; the live read moved 0 → 255 usable.
- **Evidence stores are never allowed to cost the thing they record** (R10.B-H). The Focus membership stream, the tracker transition ledger, the regime-shift rows and the journal all fail quiet: a failed append loses the event, never the pick, the tracker save, the regime change or the trade. The one exception is a journal WRITE, which reports failure loudly - a capture that did not reach disk must never look like one that did. Ground rule 10's statistics contract lives once in `scripts/evidence_stats.py` and every ground-rule-11 surface reads it from there; `outcome_semantics.claim_kind` decides what may be averaged as a trade at all, and **59% of the outcome store is annotations** that were previously averaged as trades.
- Chart paint lines (A4, landed on `testing` 2026-08-09): `scripts/chart_levels.py` builds the D1 S/R stores, prev-day H/L and the projected D1 trendline into a `levels` payload on the ChartDataService **worker** — never the paint path — and `CandleChart.set_levels` draws them with stable ids and click-to-select (`levelSelected`). One paint-lines control (`ui/widgets/paint_lines_button.py`) shows/hides groups, machine-local, defaults all-on. Trendline availability is surveyed in `docs/D1_TRENDLINE_SURVEY.md`; measure it on the desk with `scripts/d1_trendline_survey.py`.
- Price alerts: the Focus tab and Research advanced view share one `PriceAlertService`; the desk polls and pushes fired alerts to the phone at ntfy `urgent`. The satellite relay/toast layer and the planned satellite edit intents went with Desk Link (removed 2026-08-24). The generic `read_only` mode on the price-alert board and panel SURVIVES the removal — it is a widget capability with its own tests, not satellite plumbing — and now has no production caller. See `docs/FOCUS_PRICE_ALERTS_PROPOSAL.md` and `docs/EVENING_MODE_RUNBOOK.md`.
- Phone push policy (trader rule 2026-08-11, extended 2026-08-14): **AWAY is the only Auto mode that pushes routine output**, with **two** deliberate exceptions — the Research/Focus price alerts (every mode, including OFF) and EVENING's SPY ±1% wake alarm (`_maybe_push_spy_alarm`, urgent, repeating every 5 min while it holds, kill switch `push_evening_spy_alarm`). In AWAY the hourly swing push also carries the full favorite/high-conviction roster, and a second hourly push names the D1 level/event alerts since the previous one (the Alert Center classifies, `AutopilotService` aggregates and gates). Before adding any new ntfy sender, gate it on `auto_mode == AWAY` or state why it belongs with those two exceptions. The Price Alerts panel's **Test wake alert (urgent)** button (`PriceAlertService.test_push(urgent=True)`, added 2026-08-20) is a channel TEST, not a third sender: nothing schedules it, only that button calls it. It exists because ntfy has no Apple critical-alert entitlement, so urgent priority alone cannot override iOS Sleep Focus — the device-side steps are the Sleep breakthrough checklist in `docs/EVENING_MODE_RUNBOOK.md`.
- **The adoption gate compares timestamps at one seam** (`_gate_moment` in `autopilot_core`): the stored `gate_bar_end` is ALWAYS timezone-aware (it is the profile's `as_of`) while `gate_checked_at` and the caller's clock are naive `datetime.now()`. That mismatch crashed every adoption on 2026-08-19 and cost a whole session. Normalize by ATTACHING market-local to the naive side (`normalize_market_local_datetime`), never by stripping the aware side — stripping ends the crash and keeps the outage.
- **Movers-only chart review** (trader rule 2026-08-19, evening): a long inside yesterday's range is chop, so the chart-review queue shows only longs above the previous day's high and shorts below the previous day's low. It is a DEFAULT-ON PRESENTATION filter in `AlertCenterPanel._enqueue_review_alert`: it hides and counts ("N hidden (inside yesterday's range) - show", one click reveals for the session), never deletes, never mutes, never writes `review_policy.json` and never feeds the review-learning stream. UNKNOWN **shows**, tagged `unmeasured`. The deliberate Focus review (`review_focus_picks`) and armed chart-watch hits bypass it entirely. Focus surfaces flag a mover with the existing badge idiom (`MOVING`). The measurement is `focus_adoption_gate.mover_state` — the adoption gate's own extreme leg, no second copy of the rule. **Since 2026-08-27 (trader rule 2) it has BOTH legs and is asked at SHOW time:** `vwap_state` is the gate's `session_vwap_state` over `regime_pause_hold.session_levels` (cached M5, completed bars, never BounceBot's dynamic VWAP), `_review_chart_state` hides on EITHER verified leg (deliberately not the gate's UNKNOWN-before-CLOSED ordering - one measured reason to hide is enough), and `_advance_review_queue` re-measures the next candidate before showing it, because EPD reached the pane an hour after its flag under VWAP and fading. The revealed-for-the-session flag switches both checks off; a revealed name is badged `wrong side of VWAP`. **Rule 3 (same day) added a third leg for D1 recommendations only** (`is_d1` rows and `focus_d1_event` flags, never intraday): a long must be above its SMA200, a short below its SMA50 - `scripts/sma_trend_gate.py`, averages off COMPLETED daily closes (a preview / today-dated bar is excluded), price off the last completed M5 bar or else the last daily bar, badge `wrong side of SMA`. The scanner still emits trend-contrary D1 shorts; gating them at the source is a detector change and needs golden fixtures first.
- **Intraday alerts are a list beside the chart, not a queue in front of it** (trader rule 2026-08-27): `ui/widgets/m5_alert_bar.py` is the LEFT column of the desk, before the chart column (`TradingDeskPanel` splitter `bar | alert_center | setups`, three `desk_layout.DESK_SPLIT_*` weights; built in the middle and moved left the same morning on the trader's second pass, `DESK_SPLIT_KEY` bumped to v3), one line per alert newest first, `Copy all` (tickers, one per line, each once - a TC2000 paste) and `Clear all` (screen only), a click charts through `AlertCenterPanel.chart_alert` and removes the line (looked-at is done; the feed and History keep it). **Clicking from one row to the next is a SKIP, never a re-queue** (third pass, same day): `_select_review_alert` used to push every outgoing chart to the head of the waiting list, which refilled the D1 queue with the M5 rows the bar exists to keep out of it. `_current_review_holds_place` records whether the chart in front was POPPED off the queue (or is a clicked D1 row / armed hit) or merely clicked off the bar - a flag, not a re-test of the outgoing alert, because the same-symbol refresh branch swaps a queued D1 chart's alert object for that symbol's newer M5 one and `_is_m5_review_alert` would then drop a real queue member. Only a place-holder is re-inserted; the other writes a `skip` review event (dwell + `detail.reason = clicked_away_from_m5_alert`), because `_render_current_review` already emitted the `shown` impression and `shown` is the denominator for P(take | shown) - an unanswered impression would bias the rate. No parking: that stays specific to Skip-after-arming-a-D1. The routing is `_is_m5_review_alert` inside `_enqueue_review_alert` AFTER the AWAY branch - an ordinary intraday alert is emitted on `m5AlertPosted` and never queued; D1 rows, Focus D1 flags, chart-watch hits, armed price alerts, auto-pick proposals, typed symbols and the deliberate Focus review keep their chart. Everything upstream of that door (backing list, feed, History, evidence, AWAY recap) is untouched; the review queue is D1-only in practice. The queue-mechanics test files switch the routing off through one autouse fixture; `tests/test_qt_m5_alert_bar.py` owns it. **A click away IS a pass, and that is the intended meaning** (trader decision 2026-09-01, confirming the 2026-08-27 mechanic rather than changing it): *"clicking away = a pass. The tabs under the visual chart review should give us all the tools we need and we decide as we see. set alerts / add to focus and then move on."* So the `skip` row with `detail.reason = clicked_away_from_m5_alert` is not a shortfall to be repaired into a "take" or into silence by some later packet - it is the trader's answer to the impression, recorded. **The reason string is frozen**: `review_learning` keys on it, so renaming it silently re-partitions every cohort already accruing forward returns. Anything the trader wanted to keep from that chart they take with the tabs UNDER it (arm an alert, add to Focus) before moving on; those are separate writes and a pass never undoes them. Nothing in the code changed for this decision - only this entry and a one-line pointer at the writer.
- **The group RS/RW tape owns its own clock and reads 90 | 60 | 30 minutes** (rebuilt 2026-08-27 to `docs/prompts/GROUP_TAPE_REBUILD_OPUS_PROMPT.md`, plan.md Phase 0.5 item 11; hidden earlier that day and shown again by the rebuild, live gate owed). It was never wrong, it was LATE: it refreshed only when a scan cycle's RRS pass finished (10-30 min apart, once 31 minutes late on a flip) and its one intraday number was a 60-minute `real_relative_strength` window off a 5-day fetch, so for the first hour it carried the overnight gap. Now: `scripts/group_rrs.py` is the formula, PURE and lifted out unchanged - a parity test feeds identical bars to it and to `legacy.real_relative_strength` and asserts 1e-9, and `SECTOR_ETFS` is a drift-tested COPY of legacy's map so the tape survives BounceBot being off. Its session filter is `completed_bars.completed_m5_bars` **plus a same-date filter**, which is the thing that stops a window reaching over the gap; `align_bars` intersects the two series on normalized stamps so a halted ETF cannot read as strength; 6/12/18 bars = 30/60/90 run off ONE filtered+aligned series and a window without `length + 2` bars is `None`. `ui/services/group_tape_service.py` is the Strength Board's shape - one QTimer, single-flight worker, last-good on failure, bounded shutdown - doing **ONE batched `yfinance` `period=1d interval=5m` download per 5-minute tick with no retry inside the tick** (Yahoo rate-limits bursts), quiet-hours gated with `refresh_now` exempt, **zero IB traffic and no `legacy.py` change**. A missing industry map means sectors only and no SPY bars today means "no read", both said out loud. On the strip an unmeasured window draws NOTHING (0.0 would claim "in line with SPY"), chips DIFF keyed by ETF and their variants live in `theme.qss` on a `side` property, and the callout carries the as-of and the status so a stale read is never silent. **The RS Window tab and `focus_picks_panel` still read `rrsSnapshotChanged` and must keep doing so** - it answers a different question (who led over the selected window at scan time). Not built, deliberately: industry as median member return (needs member bars - an IB-budget question).
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

## Headline statistics, long form (moved verbatim from CLAUDE.md on 2026-09-03, F1 docs packet)

`CLAUDE.md` keeps the rules of this block; this is the block as it stood, with every
measurement and the reasoning behind each rule.

**Headline statistics and the priority switch (V3, decision 0016)**

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
