# GUI fluidity: how to measure a session, and what to do with the answer

**Status: active runbook.** Re-run this after any session where the desk felt
sticky, and after any change meant to make it smoother.

The trader's standing goal: *"I want this program to be very fluid to use."*
This file exists so that goal is checked against numbers instead of impressions,
by whoever is at the desk, without re-deriving the method.

---

## 1. Run it

The desk must have the watchdog on (`ui_stall_watchdog: true` in
`local_settings.json`, or `TRADINGBOTV3_UI_STALL_WATCHDOG=1`). It writes to
`%LOCALAPPDATA%\TradingBotV3\diagnostics\ui_stalls.jsonl`.

```powershell
# One command. Add --compare to hold it against a previous session.
.venv\Scripts\python.exe scripts\ui\stall_watchdog.py `
    --compare "$env:LOCALAPPDATA\TradingBotV3\diagnostics\ui_stalls_prefluidity_2026-08-21.jsonl"
```

It prints three things:

1. **the session summary** - stalls, median, p90, worst, total seconds blocked;
2. **the offender table** - per frame, total / worst / median / count;
3. **the histograms** - for the worst stalls, where their samples actually
   landed. This is the one that names a cause rather than a symptom.

**Archive the log before a fresh measurement**, or you are comparing a session
against itself plus its own history:

```powershell
$d = "$env:LOCALAPPDATA\TradingBotV3\diagnostics"
Move-Item "$d\ui_stalls.jsonl" "$d\ui_stalls_$(Get-Date -f yyyy-MM-dd_HHmm).jsonl"
```

---

## 2. The baseline to beat

Measured on the live desk, 2026-08-21 07:52-11:11, **before** the fluidity pass
(`dc75418`) and before the GC deadline fix (`ab219b5`) was running:

| | value |
|---|---|
| stalls over 50 ms | **1843** in 3h20m |
| median | **238 ms** |
| p90 | **1.16 s** |
| total blocked | **1008 s** (~8% of the session) |
| plus | GC freezes of **298 s** and **200 s** |

The wider archived log (`ui_stalls_prefluidity_2026-08-21.jsonl`, 23:23 → 11:33,
spanning the previous night too) reads **3577 stalls / 11889 s blocked** - use
the 3h20m figures above for a like-for-like session comparison.

### Targets

- **no stall over 5 s**
- **under ~60 s blocked** across a full session
- working set under ~2 GB after three hours (it reached **8.1 GB** before the GC
  deadline fix)
- **zero `QFont::setPointSizeF` lines** in the console

---

## 3. What was already fixed, so you know what is being tested

`dc75418` (the fluidity pass) and `ab219b5` (the GC deadline). In order of
measured cost:

| change | measured |
|---|---|
| Alert rows: 7 per-widget stylesheets → 0; variants are `theme.qss` rules on `alertKind`/`focusOn` | 250 rows **282 → 167 ms** |
| `FocusSideEditor.refresh` diffs instead of destroying and rebuilding every chip | no rebuild on an unchanged board |
| `ChartDataService.cached_bar_dicts` memoizes `as_bar_dicts` per series | ~490 dicts/symbol/poll → once per series |
| `project_paths._load_local_settings` mtime-cached | 100 reads **9.6 → 0.7 ms** |
| `review_events.load_review_events` stamp-cached | 5.8 MB / 8809 rows, **80.8 → 7.7 ms** |
| `setup_delegate._resized` scales in the font's own unit | kills the `QFont` console flood |
| GC deadline (`_GuiGcController`) | activity may delay a sweep, never cancel it |

**The server is not a factor and does not need re-testing.** Every hot path
resolves to `C:\TradingBotData` or `%LOCALAPPDATA%`; the GUI holds no reference
to the research store outside two worker-thread tiles; `\\MINI-PC\Trading Bot
Data` was momentarily unreachable at the time and a miss on it costs **0.0 ms**. It resolved again the same afternoon: the share drops and re-establishes, which matters for the overnight AI-store and warehouse writes but never for the GUI.

---

## 4. The next target, already located

The first post-fix run (2026-08-21 12:12-12:19, a cold start) produced a
**11,970 ms** stall whose histogram is unambiguous:

```
   210 samples  scripts/ui/panels/focus_picks_panel.py:419
     7 samples  scripts/ui/panels/alert_center_panel.py:1632
     6 samples  scripts/ui/app.py:1187
```

Line 419 is `chip.update_state(self.live_state_for(symbol, self.side))` - so the
cost is **`live_state_for`**, not the widget work the pass just fixed. The stack
names the driver:

```
focus_picks_panel.py:227  record_bounce_alert
focus_picks_panel.py:214  _refresh_all
focus_picks_panel.py:419  refresh
```

**Every bounce alert refreshes all four editors, and each one resolves
`mover_state` per symbol.** `FocusPicksPanel._mover_state_for` delegates to
`AlertCenterPanel.mover_state`, which reads `_m5_bars_for` / `_d1_bars_for` on
demand - the D1 half is memoized now, the M5 half is not, and the prev-day
computation repeats per symbol per refresh.

Two candidate fixes, in order of preference:

1. **Resolve mover state once per refresh, not once per chip** - `_refresh_all`
   builds a `{symbol: state}` map and hands it down. One pass over the Focus
   list instead of one per editor per chip.
2. **Cache `mover_state` per (symbol, side) against the M5 series identity**, the
   same shape as `ChartDataService.cached_bar_dicts`.

Neither touches a detector, a score or an alert decision - `mover_state` already
delegates to `focus_adoption_gate.mover_state`, which stays the one definition.

---

## 5. Reading a result

- **A frame with a big `total` but a small `worst`** is a papercut: frequent,
  cheap, felt as "sticky". Cache it or move it off the poll.
- **A frame with a huge `worst`** is a freeze. Find what makes it big - the
  histogram tells you whether the time is in that frame or in something it calls.
- **`app.py:<the exec line>` as the modal frame** means the main thread was in
  Qt's own C++ with no Python below it: painting, layout, or stylesheet work.
  Look for per-widget `setStyleSheet` and for lists that rebuild rather than
  diff.
- **A stall with one sample** is at the resolution limit - do not over-read a
  single sample's frame.

## 6. What cannot be checked from a test run

Qt writes its warnings straight to stderr from C, and **they do not reach a
piped stderr on Windows** - a canary `qWarning` in the test harness prints
nothing. So the `QFont` fix cannot be verified anywhere except a real console.
Look at the desk's own console window; `install_qt_message_rate_limit` also
prints a tally of any repeated Qt message when the app exits.

## 7. The desk workload bench (G0)

Sections 1-6 measure a SESSION on the live desk: they say the GUI thread was
blocked, and they cannot say which click paid for it or be run twice over the
same work. `scripts/ui/desk_bench.py` is the other tool. It builds the pages one
at a time - **never `MainWindow`**, so no IB, no autopilot and no timers - drives
a fixed workload over a staged copy of the home folder, and prints two tables:
operations and layout fit. It measures and changes nothing.

### Stage first, always

The bench never runs against the live store, and it will exit 2 rather than try.
Stage a scratch copy:

```powershell
.venv\Scripts\python.exe scripts\ui\desk_bench.py stage `
    --from "C:\TradingBotData" --to "$env:TEMP\deskbench_home"
```

It copies an ALLOWLIST of the read inputs the pages open, with the source opened
read-only, and prints bytes per file. The 1.2 GB tracker JSON, the 622 MB
attributes CSV and the 142 MB scenarios CSV are deliberately NOT on it - no page
on the bench opens them. An allowlist entry the source does not have is printed
as `absent`, never an error: the control and study discovery exports do not
exist until the next persisted tracker write. A `--to` under `C:\TradingBotData`
or the DAS is refused twice over, by two guards comparing their own literals -
because proving a guard bites means breaking it and re-running, and the first
such run here staged six files into `C:\TradingBotData\scratch` before the
second guard existed.

**`local_settings.json` is not on the allowlist, on purpose.** It was, named
against the home-folder root, where that file does not live: the real one is
machine-local at `%LOCALAPPDATA%\TradingBotV3\local_settings.json`, so the entry
printed `absent` on every staging run and the first baseline was measured on a
synthetic one-key settings file. The bench now seeds its own redirected settings
from the real machine-local file at RUN time (`machine_settings_seed`), key by
key from an allowlist: the display keys that change what is measured -
`qt_ui_scale` scales every `theme.px` and therefore every fit number -
`gui_mode`, `qt_theme`, `qt_compact_density`, the split sizes, plus the trader's
`daily_bars_source` pin. It never copies one of the five credentials or the
three live-store path keys (`shared_data_dir`, `research_store_dir`,
`ai_store_dir`) that file also holds, and `qt_autopilot_auto_arm` is forced
False rather than read. The run prints `machine settings: carried from <path>`,
or names why it could not read it and runs on the defaults.

### Run it

```powershell
.venv\Scripts\python.exe scripts\ui\desk_bench.py `
    --data-dir "$env:TEMP\deskbench_home" `
    --sizes 3456x2160 3840x2160 2560x1440 --repeat 3
```

`--data-dir` is REQUIRED and goes into `TRADINGBOTV3_DATA_DIR` **before the
first import of anything under `scripts/`**; `LOCALAPPDATA` moves into the
scratch too, so a panel that writes a cache cannot reach the machine-local store
either. **`--data-dir` and `--out` are both checked by both guards BEFORE
anything is created**, because preparing the environment is itself a write: the
first version refused a live `--data-dir` only after it had already made the
directory and written a settings file inside it. The resolved
`project_paths.DATA_DIR` is then printed, and the process exits 2 if it lands
under the live home folder or the DAS. Output
lands at `%LOCALAPPDATA%\TradingBotV3\diagnostics\desk_bench_<stamp>.json`
(`--out` overrides); `--platform windows` runs on a real screen and additionally
records each `QScreen`'s geometry and `devicePixelRatio`.

### Reading the operations table

Three numbers per op, because they answer different questions:

| column | what it is | what a bad value means |
|---|---|---|
| `sync` | the wall time of the call on the Qt thread | the click did not return |
| `settle` | until the page's workers finished and the queued renders drained | the click returned and the page kept filling in |
| `stall` | the LONGEST single `processEvents()` inside the settle wait | one slot ran that long in one go - this is the stall proxy |

`dl` counts settle deadlines. **A deadline is a RESULT, not an error**: a page
that never finished is reported as one, and the bench never sleeps to make a
number look better. `*` marks an op over 250 ms sync p95.

**Every settle carries a 120 ms floor** (`QUIET_MS`): settling is declared only
after the page has been quiet that long, so a settle reading near 120-135 ms
measured NOTHING - the op was finished before the first drain - and that op is
read on `sync` instead. The 2 ms yield the loop takes while a worker is running
(so the worker gets the GIL rather than the bench spinning against it) is inside
`settle` and outside `stall`, which brackets the `processEvents()` call alone.
**The 250 ms mark is a label, not a verdict**: three ops crossed it at the
target size in the 2026-09-06 baseline and four in the reviewer's re-run of the
same branch. Compare the ORDER of the slowest ops between two runs, not who is
over the line.

### Reading the fit table

For every page, every Weekend step and every Research child, at each size:
`minimumSizeHint`, `minimumSize` and `sizeHint` against the available height -
the window height minus chrome **measured from the widgets** (a `TopBar` frame
and a `QStatusBar` built from the same theme constants `ui/app.py` uses), not a
constant. `required` is the larger of the hint and any explicit floor. `floors`
is the sum of the explicit minimum heights of every table on the page, counting
the ones behind a tab: that floor is what the page needs the moment the trader
clicks that tab.

### Compare

```powershell
.venv\Scripts\python.exe scripts\ui\desk_bench.py `
    --data-dir "$env:TEMP\deskbench_home" `
    --compare "$env:LOCALAPPDATA\TradingBotV3\diagnostics\desk_bench_baseline_2026-09-06.json"
```

Positive deltas are SLOWER. An op that appears or disappears is named rather
than dropped, which is why the op names never contain data-dependent text (the
Weekend rail's step glyph flips from `○` to `✓`, so the ops are keyed on the
step IDs instead).

### The G0 baseline, 2026-09-06

`desk_bench_baseline_2026-09-06.json`, offscreen, three sizes, `--repeat 3`,
against a 383.6 MB staged copy of the live store. Chrome measured 90 px, so
2160 gives 2,070 px of page and 1440 gives 1,350.

Seven ops over 250 ms sync p95, across the three sizes:

| op | sync p95 (3456x2160 / 3840x2160 / 2560x1440) |
|---|---|
| `research.construct` | **5,142 / 4,312 / 4,400 ms** |
| `setup_tracker.refresh` | 1,320 / 1,098 / 1,060 ms |
| `market_journal.construct` | 299 ms (3456x2160 only) |

`weekend.refresh_everything` returns in 1.7 ms and settles in 12.7 s p50, hitting
the 20 s deadline once. The worst single `processEvents()` in the whole run was
**806.6 ms**, in `research.construct` at 2560x1440. Two pages flagged `overflow`
at every size:
`weekend_prep` (needs 3,072 px) and `weekend_prep.focus_review` (needs 2,858 px,
of which 2,340 is nine table floors of 260 px each) - the defect G1 fixes, and
the proof this check sees it. G1-G7 are re-measured against this file.

**Not offline.** Constructing the Research tab reaches
`treasury_calendar_service`, which attempted an HTTPS call on every run and
failed on certificate verification. Harmless to the numbers, but the bench is
not hermetic and must not be described as such.

### The G7 re-measure, 2026-09-07

G7 changed **when** and **on which thread** a read happens, and nothing else.
Both runs below are offscreen, `--sizes 3456x2160 --repeat 3`, against the same
385.0 MB staged copy, on the same machine, minutes apart: the BEFORE run is the
branch's base `0a1478e0` with only the seven source files under test reverted,
so the two differ by the packet and nothing else. Chrome measured 90 px.

**The bench's own overhead is now a number.** `settle` returns a fourth value,
`poll_cost_ms`, and every op row in the JSON carries its `n/p50/p95/max`; the
ops table prints one footer line, `bench poll cost (inside settle): worst
91.5 ms over 71 op(s)` - that worst case belongs to the 9.4 s
`weekend.refresh_everything` settle, so it is under 1 % of the wait it is
inside. Before G7.0 the probe walked the whole widget tree twice per poll.

| op (3456x2160) | sync p95 before | sync p95 after | settle p95 before | settle p95 after |
|---|---|---|---|---|
| `research.construct` | **3,025.1 ms** | **533.4 ms** | 6,277.1 ms | 134.2 ms |
| `setup_tracker.refresh` | 576.9 ms | **0.2 ms** | 256.1 ms | 246.8 ms |
| `market_journal.construct` | 194.7 ms (p50 57.9) | 163.1 ms (p50 9.8) | 125.9 ms | 121.5 ms |
| `research.tab.Day Trade Tracker` | 2.3 ms | **448.8 ms** | 162.8 ms | 5,929.9 ms |
| `research.tab.Setup Tracker` | 3.9 ms | 1.7 ms | 152.7 ms | 1,180.8 ms |
| `research.tab.Master AVWAP Market Prep` | 1.8 ms | 79.8 ms | 138.1 ms | 135.1 ms |
| `research.tab.Setup Playbook` | 4.6 ms | 4.8 ms | 134.3 ms | 291.3 ms |

**Read the last four rows with the first three.** Nothing got faster by being
skipped: a page's first load is now paid by the tab that asks for it, so the
cost that used to sit inside `research.construct` at startup - for eight tabs
the trader had not opened - appears against the tab they open. The whole of
`research.construct`'s 2.5 s saving is that move plus the tracker's read leaving
the Qt thread; `setup_tracker.refresh` is 0.2 ms because it now starts a
`ReadWorker` and returns, and the 246.8 ms settle is where the read is.

Two ops remain over the 250 ms sync mark, against two before:
`research.construct` (533.4 ms - constructing nine child widgets, no read) and
`research.tab.Day Trade Tracker` (448.8 ms - `reload_from_disk()` is still
synchronous on the Qt thread; G7 moved WHEN it runs, not where, and moving it to
a worker is a later packet). **The layout-fit table is byte-identical across the
two runs**, all 19 rows, which is what a lane that moves no widget should look
like.
