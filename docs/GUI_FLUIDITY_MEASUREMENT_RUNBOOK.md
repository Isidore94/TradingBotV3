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
Data` was not even mounted, and a miss on it costs **0.0 ms**.

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
