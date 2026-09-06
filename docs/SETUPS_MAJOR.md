# How the major setups work — AI-stated understanding

Document role: **active trader-review reference**, not a status or roadmap file.

This is the AI's statement of how the production ("major") setups work, written from
the code — chiefly `scripts/setup_docs.py` (the setup encyclopedia, which is kept in
sync with the scanner/tracker), `scripts/master_avwap_lib/`, and
`scripts/bounce_bot_lib/`. **Purpose: Aaron reviews this, corrects anything wrong or
mis-emphasized, and the corrections get folded back in** so every future agent starts
from a verified understanding. Companion doc: `SETUPS_TEST.md` (study/research setups).

Anything marked ⚠ is a point I'm least sure of — please check those first.

## Shared machinery (applies to every swing family)

- **Bands**: anchored VWAP from the current earnings anchor (AVWAPE) with ±1/2/3
  running-deviation σ bands (`calc_anchored_vwap_bands` — the σ formula is frozen,
  CLAUDE.md invariant). Fresh earnings anchors are skipped ~2 weeks to avoid gap
  ambiguity. The zone between AVWAPE and UPPER_1 (longs) is the "favorite zone" —
  the measured best-edge zone.
- **Signal vocabulary**: current-bar events per level — `CROSS_UP/DOWN_{AVWAPE,VWAP,
  UPPER_1..3,LOWER_1..3}` and `BOUNCE_*` (low tags the level, close recovers).
  Semantic tags derived per row by `master_avwap_lib/setup_tagging.py`
  (`setup_tags_v2`, side-consistency checked).
- **Quality gates** that add score: retest-followthrough flag (break → pull back →
  hold → resume), 5-day-high breakout, previous-day range break (Auto picks *require*
  PDH/PDL break), VWAP-range confirmation. Score caps under first-dev chop or severe
  compression.
- **Ranking**: static quality points → prior expected R, blended with the tracker's
  live regime-conditioned realized R (`master_avwap_lib/expected_r.py`). Default mode
  is "tracker leads": weight = n/(n+3) closed samples, so even ~3 recent trades pull
  the estimate halfway to realized performance; `rank_score` decays with staleness so
  points can't keep a stale signal on top. Signal weights were rebalanced 2026-07-01
  against 60-day measurements (several halved — noted per family below).
- **Buckets**: rows land in priority buckets; `favorite_setup` and `high_conviction`
  are the alert-worthy targets. D1 Focus Alerts fire only on genuine upgrades *into*
  those buckets (favorite→favorite re-flags suppressed;
  `master_avwap_bucket_state.py`).
- **Exit discipline** (the tracker's, from `setup_docs.py`):
  - Stops are **levels, not ticks**: a stop "at LOWER_1" fires after **2 daily closes**
    beyond the level (**1 close** for post-earnings setups), never on an intraday wick.
  - Default profit plan: take 50% at the 2nd favorable band, run the rest toward the
    3rd, trail the stop to the 1st band after the partial.
  - Time stop: 18 sessions if neither stop nor target resolves it.
  - Long stops sit at LOWER_1 (shorts UPPER_1); a **first-band bounce entry stops at
    AVWAPE** — one level beyond the bounced band, not two (changed 2026-07-01).

## Main swing families

- **AVWAPE → 1st Dev ("the Favorite")** — the bread-and-butter: reclaim/bounce off
  AVWAPE working through the favorite zone toward UPPER_1. Best variants carry
  retest-followthrough + previous-day range break. Entry on signal close or next open.
- **AVWAP Retest Followthrough** — the retest variant of the favorite (break, pull
  back to the level, hold, resume); entry is the retest-hold close, not the original
  break. Retest entries beat chase entries in both measurement systems.
- **AVWAP Breakout** — momentum variant (CROSS_UP without waiting for the retest);
  measurably weaker than the retest, and UPPER_2/UPPER_3 crossings barely score at all
  (they measured *negative*; rebalanced 2026-07-01).
- **AVWAP Band Bounce** — pullback tags a band intraday, close recovers. Confirmation
  is the close-back-above — never anticipate mid-bar. **Stop placement is this
  setup's edge**: one band beyond the bounced level; under-bar tick stops destroyed
  bounce entries in the playbook backfill (-1.5R) while unstopped bounces have the
  best 5-session returns (+3.4% for BOUNCE_VWAP).
- **Extreme Move Retest** — after a displacement bar (multi-band traversal, band-width
  expansion), buy/sell the first controlled pullback to the stored retest level, with
  the move.
- **SMA Breakout + Retest** — reclaim of daily SMA 50/100/200 after time below, then
  entry on the retest-confirmation bar (higher high or PDH break), not the raw
  reclaim. SMA200 reclaims were the correction-regime standout (+0.90R).
- **TOP Weekly Leader** — weekly screen (13w/26w gains, shallow pullback from 52w
  high, weekly EMA15 held, above weekly SMA100); the *daily trigger* recorded per name
  (SMA50 reclaim/bounce, weekly SMA50 retest, AVWAPE retest) is the trade — the
  pattern is only context. Leaders trend; trail rather than cap at band 3.
- **Favorite Zone Watch** — not a setup: zone residency without a trigger, kept on the
  board so the first real signal isn't missed (zone location alone measured ~0 edge).
- **General / Untagged** — fallback bucket; inspect the row's signals.

## Earnings-cycle families

- **Post-Earnings Candle Break** — *the trader's flagship post-earnings play*
  (added 2026-07-31 from live examples: VFC short, NEOG long, CAKE flag, MMM short).
  Requirements: post-earnings anchor active, gap ≥ 1.0 ATR in trade direction,
  earnings-candle color aligned with the gap; the signal is the first intraday tag
  through that candle's low (short) / high (long); a close through it adds
  CLOSE_CONFIRM (higher conviction). Stop: the candle's other extreme; pre-earnings
  AVWAPE is the wider invalidation. No 52w requirement — never stacks with the 52w
  variant. No tracker history yet; weight parked at 75.
- **Post-Earnings 52w Break** — post-earnings trend breaks the 52-week extreme.
  Highest weight historically (150) but measured ~0 edge over 60d → halved. The
  playbook says the *gap-hold* variant in weekly-strong names is the real edge
  (+0.54R, n=73).
- **Post-Earnings AVWAPE Bounce** — first pullback to the post-earnings anchor.
  Measured negative at every horizon (weight 128→64); backfill says the first tag
  works better as a SHORT in weekly-weak names. Longs: watch-and-confirm only.
- **Mid-Earnings EMA15 / EMA21 / 1st-Dev Retests** — all require a completed/active
  2nd-stdev-zone episode first, then the pullback retest (EMA15 shallow → 1st-dev
  deepest). All three measured poorly in the 60d window (weights halved / cut; the
  1st-dev variant was worst at -4.5%). **The zone streak is the edge** (10+ sessions
  beyond band 2: +8.8% at 10 sessions) — the retest entries need the full
  confirmation stack.
- Post-earnings setups use the tighter **1-close** stop-failure discipline.

## Intraday major setups (BounceBot, M5)

- ~20 bounce/trigger types on completed 5-minute bars (`bounce_bot_lib/legacy.py`
  `BOUNCE_TYPE_DEFAULTS`/`LABELS`): VWAP family (standard/dynamic/EOD ± 1σ bands,
  confluence, impulse-retest), EMA 8/15/21, 10-candle, previous-day high/low, plus
  H1 types (10-EMA bounce, blue-after-red, green→yellow fade), regime-pause RS/RW,
  ORB break± (30m+), and 8-EMA grind HOD/LOD.
- **Tiering** (`bounce_bot_lib/learning.py`): each live bounce gets a tier S/A/B/C/D
  from `production_r` — a blend (weight 0.5) of EOD entry quality and 60-minute
  "quick" production, because the two rank orders correlate at only 0.33 (fast
  producers look dead at EOD and vice versa). Composite dimensions: bounce_type 1.0,
  time bucket 0.4, market environment 0.4, Master-AVWAP bucket 0.6, focus 0.6. Thin
  segments get shrunk credit (n=10 counts half).
- **Mutes**: only a setup's own identity (`bounce_type`) may mute it — context
  dimensions influence tier but never veto (the old context mutes had auto-D'd every
  morning long). House rule: mute → CAUTION, never suppression.
- **PROVEN** (2026-07-09, "see the best bounces live"): a segment across
  bounce_type / bounce_combo / setup_family / swing_trait / focus with n ≥ 12,
  avg ≥ +0.45R, median ≥ 0 stamps a matching live bounce PROVEN — it upgrades and
  bypasses the Alert Center tier gate; a segment with avg ≥ +0.90R
  floors the alert at S. Proven *negatives* keep the mute veto. **PROVEN is the top
  alert class** — since 2026-09-01 it is the only one.
- **Banger — RETIRED 2026-09-01** (trader: *"not sure to be honest. We can probably
  remove this because idk what it is"*). It was a legacy top-alert class defined by a
  literal `"BANGER" in raw_text` match in the Alert Center, granting a tier-gate
  bypass, an always-sound and a repetition escalation. **Nothing in the tree ever
  emitted the token**: no detector path builds it (the regime-pause sweep is
  deliberately untiered and stamps no token), and 0 of 8,818 recorded review rows
  carried `banger=True` (`docs/archive/analysis/EVIDENCE_AUDIT_2026-08-22.md`, row D8b). The
  matcher, the bypass, the sound branch and the escalation branches are removed;
  the `banger` column stays in `trader_annotations`/review rows as a constant `False`
  so historical readers and the row shape are unchanged. The `REGIME_BANGER_*`
  constants in `bounce_bot_lib/legacy.py` are regime-pause thresholds — a different
  thing, and untouched.

## Registry identity (P7, 2026-09-01)

One line per production family, from the frozen crosswalk
(`scripts/setup_registry_v1.json`). The `setup_id@version` is the key; the
canonical id is what the warehouse stores in `setup_occurrence`. Read
`docs/RESEARCH_WAREHOUSE_ERD.md` for what the registry is and is not.

**Not authoritative yet** - `plan.md P4.1` freezes the identity graph, and until
then these rows describe what the code already believes rather than deciding
anything. Where two sources disagree about one setup, the registry records the
disagreement instead of picking a winner; those rows are in the registry's
`known_divergences` block, and three of them are about families listed here
(`PREVIOUS_AVWAPE_BOUNCE`, `SMA_BREAKOUT_WATCH`, `TOP_PATTERN_WATCH` - each has
its own canonical id but is documented under another family's page).

`supported_sides`, the timeframe roles, the exact completed-bar trigger and the
primary recipe are deliberately BLANK on every row: no source establishes them,
and a guess would read as established.

| `setup_id@version` | Canonical id | Label | Role | Exclusivity group |
|---|---|---|---|---|
| `avwape_to_first_dev@1` | AVWAPE_TO_FIRST_DEV | AVWAPE -> 1st Dev (Favorite) | TRADE_SETUP | `avwap_favorite_thesis` |
| `avwap_band_bounce@1` | AVWAP_BAND_BOUNCE | AVWAP Band Bounce | TRADE_SETUP | `avwap_band_bounce` |
| `avwap_breakout@1` | AVWAP_BREAKOUT | AVWAP Breakout | TRADE_SETUP | `avwap_breakout` |
| `avwap_retest@1` | AVWAP_RETEST | AVWAP Retest Followthrough | TRADE_SETUP | `avwap_favorite_thesis` |
| `extreme_move_retest@1` | EXTREME_MOVE_RETEST | Extreme Move Retest | TRADE_SETUP | `extreme_move_retest` |
| `favorite_zone_watch@1` | FAVORITE_ZONE_WATCH | Favorite Zone Watch | WATCH_STATE | `favorite_zone_watch` |
| `general@1` | GENERAL | General / Untagged | FALLBACK | `general` |
| `mid_earnings_ema15_retest@1` | MID_EARNINGS_EMA15_RETEST | Mid-Earnings EMA15 Retest | TRADE_SETUP | `mid_earnings_retest` |
| `mid_earnings_ema21_retest@1` | MID_EARNINGS_EMA21_RETEST | Mid-Earnings EMA21 Retest | TRADE_SETUP | `mid_earnings_retest` |
| `mid_earnings_first_dev_retest@1` | MID_EARNINGS_FIRST_DEV_RETEST | Mid-Earnings 1st-Dev Retest | TRADE_SETUP | `mid_earnings_retest` |
| `mid_earnings_second_dev_hold@1` | MID_EARNINGS_SECOND_DEV_HOLD | Mid Earnings Second Dev Hold | TRADE_SETUP | `mid_earnings_second_dev_hold` |
| `post_earnings_52w_break@1` | POST_EARNINGS_52W_BREAK | Post-Earnings 52w Break | TRADE_SETUP | `post_earnings_break` |
| `post_earnings_avwap_bounce@1` | POST_EARNINGS_AVWAP_BOUNCE | Post-Earnings AVWAPE Bounce | TRADE_SETUP | `post_earnings_avwap_bounce` |
| `post_earnings_candle_break@1` | POST_EARNINGS_CANDLE_BREAK | Post-Earnings Candle Break | TRADE_SETUP | `post_earnings_break` |
| `previous_avwape_bounce@1` | PREVIOUS_AVWAPE_BOUNCE | Previous Avwape Bounce | TRADE_SETUP | `previous_avwape_bounce` |
| `sma_breakout_confirmed@1` | SMA_BREAKOUT_CONFIRMED | SMA Breakout + Retest | TRADE_SETUP | `sma_breakout_confirmed` |
| `sma_breakout_watch@1` | SMA_BREAKOUT_WATCH | Sma Breakout Watch | WATCH_STATE | `sma_breakout_watch` |
| `top_pattern_entry@1` | TOP_PATTERN_ENTRY | TOP Weekly Leader | TRADE_SETUP | `top_pattern_entry` |
| `top_pattern_watch@1` | TOP_PATTERN_WATCH | Top Pattern Watch | WATCH_STATE | `top_pattern_watch` |

## Review notes / open questions for Aaron

1. Is "the major setup" to you specifically **AVWAPE → 1st Dev**, the
   **Post-Earnings Candle Break**, or the whole production set above? The doc leads
   with the shared machinery; happy to restructure around one flagship.
2. ~~⚠ Banger definition.~~ **ANSWERED 2026-09-01.** Banger was a legacy class with
   a matcher and no producer; removed 2026-09-01 by trader decision. PROVEN is the
   top class.
3. ⚠ The 2026-07-01 weight rebalance numbers quoted per family come from
   `setup_docs.py` "evidence" strings; confirm they're still the operative weights.
4. Anything here that reads correct-but-mis-weighted — i.e., true in code but not how
   you actually trade it — is exactly what this doc is for. Mark it.
