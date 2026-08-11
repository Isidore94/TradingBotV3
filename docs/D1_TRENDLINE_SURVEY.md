# D1 trendline accessibility survey (A4 prerequisite, 2026-08-09)

Document role: **historical measurement record** supporting the implemented A4
paint-line decision. It is not a roadmap or operator runbook.

The A4 packet asked one question before any line got painted: **is the stored
D1 trendline present often enough, fresh enough, and projectable enough to
draw honestly?** The answer decides whether A4 paints it or records a gap.

**Verdict: projectable — exactly, not approximately — but narrowly available.
Painted, gated, and never faked.** Coverage on the real desk is the one part
that could not be measured from the repo; `scripts/d1_trendline_survey.py`
measures it in one command and is the deliverable for that half.

---

## 1. Where the record lives and what it carries

`priority_trendline_candidate` (and its sibling
`priority_trendline_break_candidate`) sit per-symbol in the ai_state file,
written by `master_avwap_lib/legacy.py` at the point where priority rows are
refined. The record built at `legacy.py:17067-17131` carries:

| Field | Meaning |
|---|---|
| `slope_log_per_bar` | the line's slope in **log** price space, per bar |
| `current_line_price` | the line's value at `lookback_end` |
| `lookback_end` / `lookback_start` | the scan frame's last / first session |
| `start_date` / `end_date` | the two pivot dates the line is drawn through |
| `type` | `H-` / `L+` / `H-break` / `L-break` |
| `touch_count`, `angle_deg`, `atr_distance` | quality metadata |
| `start_idx` / `end_idx` | indices into the **scan's** frame — not portable |

## 2. Projectable? Yes, and exactly

The scan itself computes the line as `exp(y1 + slope * (i - x1))`
(`legacy.py:17049`), so in log space the line is straight and the record has
everything needed to reproduce it on any frame:

```
price(i) = current_line_price * exp(slope_log_per_bar * (i - i_end))
```

where `i_end` is the chart bar whose date equals `lookback_end`. One step of
`i` is one **trading day** on both sides, because the scan frame and the
chart both come from the same durable daily parquet store — so this is an
exact reconstruction, not a calendar approximation. `start_date` gives the
first pivot, so the line begins where it was actually anchored instead of
running off the left edge.

Two things make this fail, and both mean **no line** rather than a guess:

- **`slope_log_per_bar` absent.** A record written before that field existed
  carries only `current_line_price` — one number. There is no honest way to
  recover a slope from one price, and a flat line at that price would be a
  fabricated horizontal S/R level that nothing in the system believes in.
- **`lookback_end` names no bar on the chart.** Without the anchor bar there
  is nothing to count sessions from.

`chart_levels.trendline_level()` returns `None` in both cases (plan.md sec 5:
missing data is uncertainty, never confirmation).

## 3. Fresh enough? Only for a few days, and that is enforced

The record is frozen at the last scan that touched the symbol. A trendline
projects along its slope, so it degrades much faster than a daily SMA — which
is exactly why `d1_level_feed.py` gives it `TRENDLINE_MAX_AGE_DAYS = 5` against
the SMAs' 10. A4 reuses the same 5-session budget, measured as the gap between
the symbol's ai_state `last_trade_date` and the **last session on the chart**
(not wall-clock — the comparison that matters is "is this line as fresh as the
bars I am drawing it over?"). Past that, the line is dropped.

## 4. Present often enough? **This is the gap.**

The record is not written for every scanned symbol. `legacy.py:18935-18947`
filters to `priority_candidates` — rows with a favorite signal, a favorite
zone, a retest follow-through, an extreme-move or SMA-breakout watch, a first
deviation-break bonus, or a 5-day breakout. Of those, `find_directional_
trendline_candidate` returns a line only when it finds ≥2 touches on an
uninvalidated line on the correct side of the last close.

Two consequences the reviewer should hold onto:

1. **A looked-up symbol usually has no trendline.** The Chart Review workspace
   opens *any* name by design (`docs/CHART_REVIEW_WORKSPACE_PLAN.md` §5). A
   name that never reached priority-candidate status in the last scan has no
   record at all, so its chart simply has no trendline. That is correct
   behaviour, not a bug — but it means the trendline is a bonus on scanned
   names, not a dependable chart feature.
2. **The population fraction is now measured.** It was previously unmeasurable —
   the survey was written in a container with no Drive mount and no ai_state
   file. Coverage is a property of one file on one desk, so it is reported by a
   tool rather than guessed:

   ```
   .venv\Scripts\python.exe scripts\d1_trendline_survey.py --list 20
   ```

   **Desk measurement, 2026-08-09** (`C:\TradingBotData\data\runtime\master_avwap_ai_state.json`):

   | metric | count | share |
   | --- | ---: | ---: |
   | symbols in ai_state | 1100 | — |
   | with a trendline record | 62 | 5.6% of symbols |
   | projectable | 62 | **100% of records** |
   | fresh (≤ 5d) | 62 | **100% of records** |
   | **paintable today** | **62** | **5.6% of symbols** |

   Two readings, and they point the same way:

   - **The gates work perfectly.** Every record that exists is both projectable
     and fresh — 62/62 on each. Nothing is written that then fails to paint, so
     a line that appears on a chart is always exact. There is no gate to loosen
     and no partial-quality tier to reason about.
   - **5.6% confirms the design, it does not indict it.** The record is written
     only for priority candidates, so coverage tracks that population rather
     than the watchlist. This is the "bonus on scanned names, not a dependable
     chart feature" of point 1, quantified.

   The §4 heading calls this a gap; the measurement says it is a *bound*, not a
   defect. Because a painted line is always exact, the group stays defaulted ON:
   the earlier "leave it switched off if the number is small" contingency was
   written against the possibility of stale or unprojectable records, and that
   possibility is now measured at zero. Loosening the gates remains the wrong
   response, for the same reason as before.

## 5. What A4 does with all this

- Paints the trendline as a **sloped per-bar series** in the `levels` payload,
  so it clips to the candle range like everything else.
- Its stable id is `d1_trendline:{type}:{start_date}_{end_date}` — deliberately
  **no price**, because the projected price moves every session while the line
  through those two pivots is the same line. A price-bearing id would churn
  daily and never join against a capture row.
- Gated on: slope present, anchor bar resolvable, scan within 5 sessions.
- Never faked from stale slope data, and never substituted with a horizontal.
