# Overnight handoff — A4 paint-lines + packaging guards (2026-08-09)

Branch: **`claude/a4-paint-lines-packaging-nug5km`**, two commits on top of
`origin/testing` (59128c5). See §7 for why it is not `testing` itself.

| | |
|---|---|
| A4 paint lines | **Done, green, pushed** |
| Packaging guards | **Done, green, pushed** — spec was 5 packages behind; fixed |
| Frozen selftest | **Passes on a real freeze** (Linux); Windows build still owed |
| A5 (click-to-arm) | **Deferred** — see §4, exact diff included |
| Fence files | **Untouched.** One deferred edit recorded in §4 |

---

## 1. Trendline survey verdict

Full findings: `docs/D1_TRENDLINE_SURVEY.md`. Short version:

- **Projectable? Yes — exactly.** The ai_state record carries
  `slope_log_per_bar` (slope in *log* price space), `current_line_price` (the
  value at `lookback_end`), and both pivot dates. The scan itself builds the
  line as `exp(y1 + slope*(i-x1))`, so reproducing it on the chart is
  `current_line_price * exp(slope * (i - i_end))`. One step of `i` is one
  trading day on both sides — scan frame and chart come from the same durable
  daily store — so this is exact, not a calendar approximation.
- **Fresh enough? For five sessions.** Reused `d1_level_feed`'s
  `TRENDLINE_MAX_AGE_DAYS = 5`, measured against the chart's last session
  rather than wall-clock.
- **Present often enough? This is the gap.** `legacy.py:18935-18947` writes
  the record only for `priority_candidates` (favorite signal / zone / retest /
  extreme-move / SMA-breakout / first-dev-break / 5d-breakout), and only when
  a valid ≥2-touch line exists. **A looked-up symbol usually has none** — and
  Chart Review's whole point is looking up any name.
- **Coverage was not measurable here.** No Drive mount, no ai_state file in
  this container. That is a property of one file on one desk, so I shipped
  the measurement instead of a guess:
  `.venv\Scripts\python.exe scripts\d1_trendline_survey.py --list 20`.
  If "PAINTABLE TODAY" comes back tiny, the honest response is to leave the
  trendline group switched off — **not** to loosen the gates.
- **Nothing is faked.** No slope, or no resolvable anchor bar, or a stale
  scan ⇒ no line. A flat line at `current_line_price` would be a fabricated
  horizontal, which is worse than an absent one (plan.md sec 5).

## 2. What A4 built

- `scripts/chart_levels.py` — new, pure, no Qt. Builds the `levels` payload:
  D1 horizontal S/R (`hv_horizontal` + `cloud_flat`), prev-day H/L, the
  projected trendline. Also owns the group vocabulary and the
  `visible_overlays` / `visible_levels` filters the toggle uses.
- `CandleChart.set_levels` / `set_overlays` / `level_at` / `select_level` /
  `levelSelected` — horizontals as `pg.InfiniteLine`, sloped lines as curves,
  both `ignoreBounds=True`.
- `ui/widgets/paint_lines_button.py` + `ui/services/paint_lines_prefs.py` —
  one control, six groups, machine-local, defaults all-on.
- `ChartDataService.build_snapshots` attaches `d1["levels"]`.
- 58 new tests across `tests/test_chart_levels.py` and
  `tests/test_chart_paint_lines.py`.

### Choices I made without asking (packet said to, and note them)

1. **Styling.** Green-bucket S/R: solid, weight scaled by `level_conviction`
   (1.0–1.8). Red-bucket: faint dashed, 0.9. Cloud flats: dotted light blue.
   Prev-day: dashed white, "PDH"/"PDL". Trendline: solid purple 1.4 —
   deliberately *not* yellow, which "AVWAPE prev" already owns.
2. **Clutter budget.** Levels outside the candle price range are dropped in
   the builder (the chart's y-range follows the candles and y-panning is off,
   so they could never become visible), then capped at 10 green / 6 red / 4
   cloud by conviction. Constants at the top of `chart_levels.py`.
3. **Level build lives in `ChartDataService`, not `chart_snapshot.py`.** The
   packet said "one builder change". I put it in the service rather than the
   snapshot module because the ai_state trendline record is *also* detector
   input (`d1_level_feed` feeds the Technical Integrity monitor) and
   `chart_snapshot` is alert-adjacent (its earnings anchors feed watch
   evaluation). This keeps A4's diff off every detector/scoring/alert file.
   Cost: the retired `desk_link/popup_payload.py` path gets no levels.
4. **Id rule** (documented in `level_id()`): `family:anchor_date:price_to_the_cent`,
   preferring a store-native `id`/`level_id` field if one ever appears. A
   clustered horizontal's price is a volume-weighted mean, so a level that
   absorbs a member and shifts >1¢ gets a **new id** — the honest limit. The
   capture row also stores `ref_level_family` and the price, which is what
   analysis joins on, so a re-cluster costs a link, not the evidence.
   Trendline ids are `d1_trendline:{type}:{start}_{end}` with **no price**,
   because the projection moves daily while the line does not.
5. **A miss clears the selection.** The highlight means "this is the line the
   next capture references", so it must stop saying that on a click elsewhere.
6. **I duplicated ai_state parsing** rather than extending
   `d1_level_feed._extract_symbol_entry` (which drops slope/dates). That file
   is detector input ⇒ ask-first ⇒ not overnight. Cost: one extra parse of the
   38MB file per ai_state write, mtime-cached, on a worker. **Suggested
   follow-up (needs the ask):** give `d1_level_feed` a shared cached raw
   loader both consumers use.

## 3. What A4 deliberately did *not* do

- Did not connect `d1LevelSelected` into the capture rail's `ref_level_id` /
  `ref_level_family`. That connect lands in `alert_center_panel.py` (fenced).
  Push side (`d1LevelSelected`) and pull side (`selected_d1_level()`) both
  exist, so it is a one-line connect when approved.
- Did not touch A3. The Chart Review workspace's chart area is still the
  stated placeholder; A4's lines currently paint everywhere the D1 chart
  *already* appears (`SymbolSnapshotWidget`, so the popup and the embedded
  review pane). When A3 lands, it inherits them for free.

## 4. A5 — deferred, and why

Not blocked by the reason the packet anticipated: `PriceAlertService`'s public
API (`entries()` / `save_entries()`) is sufficient, so **no fence-file edit is
strictly required to write an alert**. I deferred it for a different reason
the packet did not cover, which is exactly the kind of ambiguity the
file-scoped ask-first rule says to escalate:

- The named pattern (`alert_chart_review.py:55-58, :87-89`) is **signal
  forwarding to a host that owns the service** — the widget never writes. The
  host on that path is `alert_center_panel.py`, **fenced**. So following the
  pattern faithfully needs a fence edit at the last mile.
- The alternative — giving `SymbolSnapshotWidget` its own service handle and
  writing `price_alerts.json` from the chart widget — would make a chart a
  **second writer of a mutable shared export**. plan.md sec 5 says one
  component owns each. That is the trader's call, not mine at 03:00.

**The deferred edit, exactly.** In `alert_chart_review.py` (not fenced) add:

```python
levelAlertRequested = Signal(str, str, float)   # symbol, direction, level
...
self.snapshot.d1LevelSelected.connect(self._on_level_selected)
...
def _on_level_selected(self, symbol, level_id, family, price):
    self._selected_level = (symbol, level_id, family, price)
```

and then the **one fenced line**, in `alert_center_panel.py` where the other
`levelArmRequested` connect already lives:

```python
self.review.levelAlertRequested.connect(self._arm_price_alert_from_level)
```

with `_arm_price_alert_from_level` doing the caller-only merge that
`price_alert_board.py:200-217` already models (read `entries()`, upsert
`above`/`below` for the symbol, `save_entries()`).

Decide the ownership question first: should a chart be allowed to write
`price_alerts.json`, or must it always go through the panel that owns it?

## 5. Packaging guards

- **`tests/test_packaging_spec_drift.py`** *executes* the spec with the
  PyInstaller API stubbed (so it asserts behaviour, not text) and checks every
  top-level `scripts/` package is collected and every non-`.py` runtime asset
  is bundled. **It failed on the first run, as predicted.** The spec was
  missing `ai_jobs`, `desk_link`, `gui_app`, `indicators`, `market_prep_gui` —
  26 submodules that would have shipped absent. Spec fixed; two documented
  allowlists (`UNBUNDLED_ASSETS` for the three operator `.ps1` files,
  `UNCOLLECTED_PACKAGES` deliberately empty).
- **`launch_gui.py --selftest`** (`scripts/selftest.py`) — 25 lazy-engine
  imports + 5 asset/behaviour checks, all failures collected and named, no
  window, no network, no QApplication. Routed before the crash log and before
  `ui.app`, pinned by a test.

### Frozen verification — what I could and could not do

**Could:** built the bundle for real (`pyinstaller packaging/tradingbotv3.spec
--noconfirm`, exit 0, 576MB) and ran the frozen binary:

```
selftest OK: 30/30 checks passed (frozen)
```

Then deleted `_internal/ui/theme.qss` out of the built bundle and re-ran:
exit **1**, naming the missing file. So it is a guard, not a rubber stamp.

**Could not:** this ran in a **Linux** container, so that is a Linux freeze,
not `TradingBotV3.exe`. The spec logic, the collection of all 11 packages, the
asset mirroring and the selftest mechanism are all proven end-to-end — but
**one Windows build is still owed** before the click-through is formally
retired:

```powershell
.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm
dist\TradingBotV3\TradingBotV3.exe --selftest
```

Unattended, ~4 min, no trader clicking. If it prints `30/30 ... (frozen)` the
click-through is done. CLAUDE.md and AGENTS.md are updated to say so.

## 6. Test baseline — read this before you panic

`.venv/bin/python -m pytest tests/ -q` in this container:
**16 failed, 2234 passed, 5 skipped, 7 subtests.**

**All 16 failures are pre-existing and environmental.** Verified by stashing
every change and re-running the same six files on a clean `origin/testing`
tree: byte-identical failure list. They are clock/timezone/market-session
dependent (`test_autopilot_core`, `test_breadth_backfill`, `test_vold_recorder`,
`test_tracker_staleness_catchup`, `test_technical_integrity`, and
`test_chart_snapshot::test_todays_stored_partial_bar_...`) and this container
has no `local_settings.json`, so `get_market_session_window` takes its
fallback branch. They should be green on the desk. **Confirm the desk's own
`pytest tests/ -q` is clean before merging** — I could not.

Two other environment notes: `pytest-timeout` is not in `requirements-dev.txt`
and one of these tests *hangs* rather than fails without it (a 25-minute wall
with 28s of CPU). And tree-wide `ruff check scripts/` reports 1639 errors —
identical count at `origin/testing`, so pre-existing, probably a newer ruff
than `constraints.txt` pins. My files are clean.

## 7. Branch note

The packet said to commit directly to `testing`. My harness instructions
pinned me to `claude/a4-paint-lines-packaging-nug5km` and forbade pushing
elsewhere, so I took the conservative route: reset that branch to
`origin/testing` (a fast-forward — its old tip 12efa71 is an ancestor) and
committed there. **It fast-forwards cleanly into `testing`**; nothing was
lost and no history diverged.

## 8. What to scrutinise first

1. **The trendline projection** (`chart_levels.trendline_level`). It is the
   only piece doing real geometry, and its correctness rests on the claim
   that the scan's frame and the chart's bars share a trading-day index. Both
   read the same durable store, but if a symbol's store were ever backfilled
   differently from the scan's frame, the line would sit at the right shape
   and the wrong place. Worth one eyeball on a real chart against TC2000.
2. **The stable-id rule's re-clustering limit** (§2.4). It is documented and
   tested for the stable case; the question is whether losing the id link on a
   re-clustered level is acceptable for Part B analysis, or whether the store
   should start carrying its own ids.
3. **Clutter on a real symbol.** The 10/6/4 budget is a guess made without a
   real level store in front of me. It may be far too many lines.
4. **The compact pane.** The "Lines" button is on the D1 legend row and I
   made it flat/22px in compact mode, but I have never seen it render at
   2560x1440. If it eats chart height in the embedded review pane, that is
   worse than the feature is worth.

## 9. The EOD / auto-away packet — hands off, as instructed

Nothing of it is on this branch and I touched none of it. `git branch -a`
shows two related remote branches that were **not** in the packet's
description and that I left completely alone:
`origin/claude/auto-away-mode-drive-output-bek3ks` and
`origin/claude/auto-evening-mode-alerts-yvg8yd`. There is no
`scripts/eod_review.py` anywhere on `testing`, and I did not open, import, or
edit `autopilot_service.py` or `autopilot_core.py`. Note that
`tests/test_autopilot_core.py` is in the pre-existing failure list in §6 —
that predates me and is unrelated to that packet.

---

## 10. Orchestration session (2026-08-09, follow-up): Phase 1 answers

Branch note first: this session worked on
`claude/a4-verify-a3-orchestrate-9c8kop`, fast-forwarded onto the A4 tip
`acf70e0` (main is an ancestor of testing; nothing discarded). Everything
below is committed there; it still fast-forwards into `testing` after the A4
branch.

### 1. Where the coding agent runs

A fresh Linux container. No Drive mount, no ai_state, no local_settings.json,
no TWS, no Windows. Worse than the previous session's container: system
Python is 3.11 (project floor is 3.12) and tkinter is absent, which kills 6
test files at collection. **Phase 1 items 2-5 are desk-only.** The desk
checklist in §12 is the deliverable for them.

### 2. Desk test baseline — NOT obtainable here; and a warning

The §6 baseline (16 failed / 2234 passed) is itself container-specific: this
session's container shows 56 failed / 1996 passed / 5 skipped / 27 errors on
the reduced suite (6 tkinter-blocked files ignored) — all environmental.
Verified by differential: the same reduced suite on a pristine A4-tip
worktree produces a byte-identical failed-ID list. **Absolute container
counts must never be used as a merge gate; only a desk run or a
same-container differential is evidence.**

Diagnosis of the §6 "hanging test": it is an **at-exit** hang — after the
summary prints, ~100 non-daemon threads from `multitasking` (a yfinance
dependency) block the interpreter's exit (the 25 min wall / 28 s CPU
signature). `pytest-timeout` (now a dev dependency) bounds *in-test* hangs
on demand but does not fix this one; runners should wrap the suite in an
outer `timeout` and judge by the printed summary. The honest fix
(daemonizing/joining yfinance workers or quarantining the import) is future
work, noted rather than patched blind.

### 3. Trendline coverage — desk checklist step 3

Tool already on the branch (`scripts/d1_trendline_survey.py`). If PAINTABLE
TODAY is near zero, the trendline group defaults to off — a design change to
escalate, not a docs edit.

### 4. Real level store — desk checklist step 4, plus a structural finding

New this session: `scripts/d1_level_store_survey.py` (commit `8a72f5c`),
read-only, reports per-symbol store census, every filter stage, and pre/post
clutter-budget counts so the guessed 10/6/4 can be checked in one command.

**Structural finding (needs desk confirmation, then a trader decision):** a
red-bucket `hv_horizontal`'s strength is bucket weight 0.35 + touch term
capped at 0.40 = **max 0.75, always below `MIN_HORIZONTAL_STRENGTH = 1.0`**
(`master_avwap_lib/levels.py` `_level_strength`). If that holds on the real
store, **no red level is ever drawn**, `MAX_RED_HORIZONTALS = 6` never
binds, and A4's red styling is dead code in practice. The options are (a)
accept that A4 draws green + cloud only and delete the red path, or (b) give
red levels their own draw threshold in `chart_levels.py` (chart-only file,
but a change to what the trader sees ⇒ ask first). Do not touch
`MIN_HORIZONTAL_STRENGTH` itself — it mirrors detector-input gating.

### 5. Frozen Windows build — desk checklist step 5

Impossible on Linux. Unchanged expectation: exit 0 +
`selftest OK: 30/30 checks passed (frozen)` retires the click-through.

### 6. Trader decisions — escalated, not decided

a. **A5 ownership** (gates A5): may a chart widget write
   `price_alerts.json`, or must arming route through the owning panel (one
   fenced connect line in `alert_center_panel.py`)? §4 above has both diffs.
b. **d1_level_feed shared cached ai_state loader** (ask-first file): approve
   or keep the duplicated parse (one extra mtime-cached parse of the 38 MB
   file per ai_state write, on a worker).
c. **A3 scope**: moot for this session — desk facts (items 2-5) were
   unobtainable in-container, so this session was verification + tooling.
   The A3 packet is drafted in §13, ready to dispatch once 6c is answered
   and the desk numbers exist.
d. *(new)* **Red-level drawing** — see §10.4 above.

## 11. What this session dispatched and what came back

| Packet | Result |
|---|---|
| pytest-timeout dependency | `ec33194`, pushed. Pin 2.4.0, no resolver movement. |
| d1_level_store_survey tool | `8a72f5c`, pushed. Fixture-verified, exit-2 failure paths clean, spec-drift test green (9 passed). |
| Regression gate for both | Differential vs pristine A4-tip worktree: failed-ID lists byte-identical ⇒ no regression. Smoke 7/7. |
| Lines-button measurement (read-only) | Themed, offscreen, 2560w and 1280w, both densities: header row 22 px with the button vs 17 px without ⇒ **5 px of chart height lost in the embedded pane; the docstring's zero-cost claim was false.** Label growth ("Lines (2 off)") costs width only — safe. |
| Lines-button height-neutral fix | `1c79d0b`, pushed. Cap = themed font line height (16 px) + `rowChrome` QSS exemption for legibility; header delta **0** at all four themed configurations; new 4-param regression test (fails 4/4 with the exemption reverted); differential suite byte-identical; smoke 7/7. |

No fence file was touched by anyone. `tests/test_packaging_spec_drift.py`
unweakened. Nothing was merged to `main` or `testing`.

## 12. Desk checklist (trader or a desk agent; ~15 min + one 4-min build)

**SAFETY FIRST:** a scheduled task runs unmerged branch code in production on
the desk (docs/CHECKPOINT_REVIEW_2026-08-08.md). **Disarm it before
switching branches; re-arm after restoring the production branch.**

```powershell
# 1. fetch + checkout + refresh dev deps (pytest-timeout is new)
git fetch origin claude/a4-verify-a3-orchestrate-9c8kop
git checkout claude/a4-verify-a3-orchestrate-9c8kop
.venv\Scripts\pip.exe install -r requirements-dev.txt -c constraints.txt

# 2. desk test baseline (Phase 1 item 2) — record counts AND $LASTEXITCODE
.venv\Scripts\python.exe -m pytest tests\ -q ; echo $LASTEXITCODE
# expected: fully green. Any failure list changes the merge gate — report it.

# 3. trendline coverage (item 3) — record all five numbers + the list
.venv\Scripts\python.exe scripts\d1_trendline_survey.py --list 20

# 4. real level store (item 4) — pick a liquid symbol from step 3's list
.venv\Scripts\python.exe scripts\d1_level_store_survey.py --symbol XXXX
# record: census, green/red split, cloud flats, range survivors, pre/post
# budget, and whether ANY red level clears strength >= 1.0 (see §10.4).

# 5. frozen Windows build (item 5) — unattended, no click-through
.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm
dist\TradingBotV3\TradingBotV3.exe --selftest ; echo $LASTEXITCODE
# expected: exit 0, "selftest OK: 30/30 checks passed (frozen)"
```

6. Eyeball (2A): launch the GUI, open a symbol from step 3's paintable list,
   and check the purple D1 trendline sits on the two pivots it claims —
   against TC2000. Wrong-but-plausible is the failure mode; report, don't
   tune. Also glance at the "Lines" button in the embedded review pane at
   2560x1440: after `1c79d0b` the header row must be no taller than it was
   without the button.

## 13. A3 packet, drafted (dispatch only after 6c + desk numbers)

- Replace the placeholder in `chart_review_panel.py::_chart_area()`
  (lines ~155-180) by embedding the existing `SymbolSnapshotWidget`
  (`symbol_snapshot_dialog.py`) — the same widget the popup and the embedded
  review pane already use. **One data path**: `ChartDataService` +
  `CandleChart` via `snapshotReady`. If an agent starts writing a second
  loader, kill the packet.
- A4 levels ride `d1["levels"]` on the snapshot — free.
- Crosshair + OHLC readout: **new work in `candle_chart.py`** (none exists
  today); shared widget, so it lands for every chart consumer — keep it
  paint-only, zero I/O.
- Wire the panel's permanent provenance strip from the snapshot's
  `meta["source"]` tier; IBKR streaming while focused with a loud yfinance
  fallback banner (degraded data must look degraded, plan.md sec 5).
- Fences, branch discipline, differential test gate: as in §11's packets.

## 14. Least confident

The at-exit-hang diagnosis is from one container. If the desk suite also
hangs after its summary, the multitasking/yfinance thread leak is confirmed
desk-side and deserves its own small packet; if the desk exits cleanly, the
hang was container-only and the note above is merely operational.
