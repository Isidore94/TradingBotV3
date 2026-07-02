#!/usr/bin/env python3
"""Setup encyclopedia: how every tracked setup works, exactly.

One structured entry per setup family: what the setup is, the exact detection
rules the scanner applies, how the entry triggers, where the stop belongs, and
the profit-take plan — written to match the code (scanner signals, tracker stop
candidates/exit templates) so the docs and the bot never drift apart silently.

Also provides ``build_trade_plan``: given a symbol's current anchored-VWAP
bands (from the scan's ai_state), it turns the family's stop/target rules into
concrete prices with R multiples. Pure stdlib on purpose — the Qt UI imports
this without dragging the scanner stack into the app process.

Ground rules shared by every setup (the tracker's exit discipline):
- Stops are LEVELS, not ticks: a stop "at LOWER_1" triggers after
  ``STOP_CLOSE_FAILURES`` daily closes beyond the level (1 close for
  post-earnings setups), not on an intraday wick.
- Default profit plan is the measured-best baseline template: take 50% at the
  2nd favorable deviation band, run the rest toward the 3rd band with the
  stop trailed to the 1st band after the partial.
- Every setup times out at ``TIME_STOP_SESSIONS`` sessions if neither stop nor
  target resolves it.
"""

from __future__ import annotations

STOP_CLOSE_FAILURES = 2
POST_EARNINGS_STOP_CLOSE_FAILURES = 1
TIME_STOP_SESSIONS = 18

_FIRST_BAND_BOUNCE_SIGNALS = {"LONG": "BOUNCE_UPPER_1", "SHORT": "BOUNCE_LOWER_1"}


def _norm_side(side: str) -> str:
    return "SHORT" if str(side or "").strip().upper() == "SHORT" else "LONG"


def protective_band_label(side: str) -> str:
    return "LOWER_1" if _norm_side(side) == "LONG" else "UPPER_1"


def favorable_band_label(side: str, band_number: int) -> str:
    prefix = "UPPER" if _norm_side(side) == "LONG" else "LOWER"
    return f"{prefix}_{max(1, min(3, int(band_number)))}"


# ---------------------------------------------------------------------------
# The encyclopedia
# ---------------------------------------------------------------------------
SETUP_DOCS: dict[str, dict] = {
    "avwape_to_1stdev": {
        "label": "AVWAPE -> 1st Dev (Favorite)",
        "group": "Main swing",
        "what": (
            "The bread-and-butter favorite: price reclaims or bounces off the earnings-anchored "
            "VWAP (AVWAPE) and works through the favorite zone toward the first deviation band. "
            "The zone between AVWAPE and UPPER_1 (longs) is where the scan measures its best "
            "sustained edge, so setups that trigger there outrank everything else."
        ),
        "detection": [
            "Anchored VWAP + stdev bands computed from the current earnings anchor (fresh earnings "
            "anchors are skipped for ~2 weeks to avoid gap ambiguity).",
            "Current-bar signal: BOUNCE_AVWAPE (low tags AVWAPE, close recovers above it) or "
            "CROSS_UP_AVWAPE / CROSS_UP_UPPER_1 (close crosses the level; prior close was below).",
            "Quality gates: retest-followthrough flag (break, pull back to the level, hold, resume), "
            "5-day-high breakout, previous-day range break, VWAP-range confirmation.",
            "Score caps if the first deviation has been chopping (repeated failed UPPER_1 pushes) or "
            "the setup sits in severe compression.",
        ],
        "entry": (
            "On the signal close, or next open. Best variants carry the retest-followthrough flag plus "
            "a previous-day range break — entering strength that already re-proved the level."
        ),
        "stop": (
            "Protective band one level below the zone: LOWER_1 for longs (UPPER_1 for shorts). If the "
            "entry was a first-band bounce (BOUNCE_UPPER_1/BOUNCE_LOWER_1), the honest stop is AVWAPE — "
            "one level beyond the bounced band, not two. Stop fires after 2 daily closes beyond the level."
        ),
        "targets": (
            "50% off at the 2nd deviation band (UPPER_2 longs / LOWER_2 shorts), remainder toward the 3rd "
            "band with the stop trailed to the 1st band after the partial. Time stop 18 sessions."
        ),
        "evidence": (
            "Tracker leaderboard (60d): BOUNCE_VWAP +3.4% and CROSS_UP_VWAP +2.4% 5-session edge; "
            "BOUNCE_UPPER_1 positive at every horizon (+2.0% at 10 sessions). Chasing crossings of the "
            "2nd/3rd band instead measured negative — wait for the retest."
        ),
    },
    "avwap_retest_followthrough": {
        "label": "AVWAP Retest Followthrough",
        "group": "Main swing",
        "what": (
            "Legacy label for the retest variant of the favorite: price breaks a band level, pulls back "
            "to it, holds, and resumes. Canonically folded into AVWAPE -> 1st Dev when the zone matches."
        ),
        "detection": [
            "A prior cross of AVWAPE/UPPER_1, then a pullback that tags the broken level within tolerance "
            "and closes back on the trend side (retest_followthrough flag).",
            "Same band/zone math and quality gates as AVWAPE -> 1st Dev.",
        ],
        "entry": "The retest-hold close (the followthrough bar), not the original break.",
        "stop": "The retested level's protective band (LOWER_1 longs / UPPER_1 shorts); AVWAPE for first-band bounces.",
        "targets": "50% at band 2, rest to band 3, trail band 1 after partial. Time stop 18 sessions.",
        "evidence": "Retest entries beat chase entries across both measurement systems (see AVWAPE -> 1st Dev).",
    },
    "avwap_breakout": {
        "label": "AVWAP Breakout",
        "group": "Main swing",
        "what": (
            "Momentum variant: close crosses up through AVWAPE or UPPER_1 without waiting for the retest. "
            "Faster but measurably weaker than the retest variant — the scoring reflects that."
        ),
        "detection": [
            "CROSS_UP_AVWAPE or CROSS_UP_UPPER_1 on the current bar (close above the level, prior close below).",
            "5-day breakout and previous-day range break add score; UPPER_2/UPPER_3 crossings barely score at "
            "all (rebalanced 2026-07-01: they measured negative).",
        ],
        "entry": "Signal close or next open; prefer names that also broke the previous day's range.",
        "stop": "Protective band (LOWER_1 longs / UPPER_1 shorts), 2-close failure discipline.",
        "targets": "50% at band 2, rest to band 3, trail band 1. Time stop 18 sessions.",
        "evidence": (
            "First-dev breakouts held small positive edge in both playbook regime windows with tight stops; "
            "second/third-band chases were the worst long signals in the tracker leaderboard (-2.8% at 5 sessions)."
        ),
    },
    "avwap_band_bounce": {
        "label": "AVWAP Band Bounce",
        "group": "Main swing",
        "what": (
            "Price trading above a band pulls back, tags it intraday, and closes back on the trend side — "
            "buying the dip at a defined level instead of chasing."
        ),
        "detection": [
            "Prior close on the trend side of the band; today's low (high for shorts) tags the level within "
            "ATR tolerance; close recovers beyond it (BOUNCE_AVWAPE / BOUNCE_UPPER_1 / BOUNCE_VWAP signals).",
        ],
        "entry": "The bounce close. Do not anticipate mid-bar — the close-back-above is the confirmation.",
        "stop": (
            "One band level beyond the BOUNCED level: AVWAPE bounce stops at LOWER_1; UPPER_1 bounce stops at "
            "AVWAPE (changed 2026-07-01 — the far protective band understated per-setup R). Never a tick stop "
            "under the bounce bar: the playbook backfill showed under-bar stops destroy bounce entries."
        ),
        "targets": "50% at band 2, rest to band 3, trail band 1. Time stop 18 sessions.",
        "evidence": (
            "Bounce entries have the strongest unstopped 5-session returns in the tracker (BOUNCE_VWAP +3.4%) "
            "but were untradeable in the playbook with tight under-bar stops (-1.5R avg) — stop placement IS "
            "this setup's edge."
        ),
    },
    "extreme_move_retest": {
        "label": "Extreme Move Retest",
        "group": "Main swing",
        "what": (
            "After a displacement bar (band-width expanding, multi-band traversal), the first controlled "
            "pullback to the expansion's reference level gets bought/sold with the move's direction."
        ),
        "detection": [
            "A recent displacement flagged by band-cross count and band-width-in-ATR (extreme_move_watch, "
            "displacement date recorded).",
            "Current bar retests the stored retest level (extreme_move_retest_level) and closes back with the move "
            "(EXTREME_MOVE_RETEST signal); favorite-ready flag requires the retest to be orderly.",
        ],
        "entry": "The retest-hold close at the stored level.",
        "stop": "Protective band beyond the retest level, 2-close discipline.",
        "targets": "50% at band 2, rest to band 3, trail band 1. Time stop 18 sessions.",
        "evidence": (
            "Positive both systems: +0.65% 5-session edge in the tracker; extreme band width 2-3 ATR was one of "
            "the highest-impact leaderboard factors (avg +34.8% at 10 sessions, small n)."
        ),
    },
    "post_earnings_52w_break": {
        "label": "Post-Earnings 52w Break",
        "group": "Earnings cycle",
        "what": (
            "An earnings gap re-anchors the tape; when the post-earnings trend then breaks the 52-week "
            "extreme, the break is traded with earnings momentum behind it."
        ),
        "detection": [
            "Post-earnings anchor active (gap date + gap size in ATR recorded; bands expanding preferred).",
            "POST_EARNINGS_52W_BREAK signal: intraday then closing break of the 52w high (low for shorts); "
            "close-confirm flag (POST_EARNINGS_CLOSE_CONFIRM) upgrades it.",
        ],
        "entry": "The closing break, or the first hold above the broken 52w level.",
        "stop": (
            "Post-earnings AVWAPE (the gap anchor's VWAP) — 1-close failure, tighter discipline than normal. "
            "Alternates the tracker also measures: the earnings candle low/high, EMA_15, EMA_8."
        ),
        "targets": "50% at band 2 of the post-earnings anchor, rest to band 3, trail band 1. Time stop 18 sessions.",
        "evidence": (
            "Was the highest-weighted signal (150) but measured ~0 edge over 60d; weight halved 2026-07-01. "
            "Playbook: post-earnings gap-hold entries in weekly-strong names were the single best combo "
            "(+0.54R avg, n=73) — the gap-hold variant beats the raw 52w-break variant."
        ),
    },
    "post_earnings_avwap_bounce": {
        "label": "Post-Earnings AVWAPE Bounce",
        "group": "Earnings cycle",
        "what": (
            "The first pullback to the post-earnings anchored VWAP after an earnings gap: institutions defending "
            "the post-earnings average price."
        ),
        "detection": [
            "Post-earnings anchor active, within the post-earnings window.",
            "POST_EARNINGS_AVWAPE_BOUNCE signal: tag of the post-earnings AVWAPE, close back on the gap side.",
        ],
        "entry": "The bounce close at the post-earnings AVWAPE.",
        "stop": (
            "Under the post-earnings AVWAPE with 1-close failure; earnings-candle low/high is the harder invalidation "
            "behind it."
        ),
        "targets": "50% at band 2, rest to band 3, trail band 1. Time stop 18 sessions.",
        "evidence": (
            "Measured negative at every horizon in the 60d window (weight 128 -> 64 in the rebalance). Backfill agrees "
            "the first tag works better as a SHORT in weekly-weak names (failed-rally variant). Treat longs as "
            "watch-and-confirm, not automatic."
        ),
    },
    "mid_earnings_ema15_retest": {
        "label": "Mid-Earnings EMA15 Retest",
        "group": "Earnings cycle",
        "what": (
            "Mid-cycle (between earnings), after the name held the 2nd-stdev zone, the pullback retest of the "
            "daily EMA15 is the re-entry into an established power trend."
        ),
        "detection": [
            "A completed or active second-stdev-zone episode (streak of closes beyond UPPER_2/LOWER_2 recorded "
            "with start/end dates).",
            "Within the post-zone window, the current bar tags EMA15 and closes back on the trend side "
            "(MID_EARNINGS_EMA15_RETEST signal; EMA8/EMA21/first-dev confluence flags add score).",
        ],
        "entry": "The EMA15 reclaim close; trade-ready needs a previous-day range break plus strength confirmation.",
        "stop": "EMA_15 level itself (2-close failure), with EMA_21 as the wider alternate the tracker measures.",
        "targets": "50% at band 2, rest to band 3, trail band 1. Time stop 18 sessions.",
        "evidence": (
            "Weights halved 2026-07-01 (measured -2.8% at 10 sessions in the 60d window). The zone-streak factor is "
            "the strong part: 10+ sessions in the zone measured +8.8% at 10 sessions — the hold is the edge, the "
            "retest entry needs the confirmation gates."
        ),
    },
    "mid_earnings_ema21_retest": {
        "label": "Mid-Earnings EMA21 Retest",
        "group": "Earnings cycle",
        "what": "Deeper-pullback sibling of the EMA15 retest: same zone prerequisite, retest of the daily EMA21.",
        "detection": [
            "Same second-stdev-zone episode prerequisite.",
            "Tag of EMA21 with a close back on the trend side (MID_EARNINGS_EMA21_RETEST signal).",
        ],
        "entry": "The EMA21 reclaim close with confirmation.",
        "stop": "EMA_21 (2-close failure); first deviation band as the wider alternate.",
        "targets": "50% at band 2, rest to band 3, trail band 1. Time stop 18 sessions.",
        "evidence": "Same story as EMA15 retest — halved weight, zone streak carries the edge.",
    },
    "mid_earnings_1stdev_retest": {
        "label": "Mid-Earnings 1st-Dev Retest",
        "group": "Earnings cycle",
        "what": "Deepest sanctioned mid-cycle pullback: retest of the first deviation band after a 2nd-stdev zone run.",
        "detection": [
            "Second-stdev-zone episode prerequisite; pullback reaches the first deviation band (UPPER_1 longs).",
            "Tag + close back beyond it (MID_EARNINGS_FIRST_DEV_RETEST signal).",
        ],
        "entry": "The first-dev reclaim close; watch-only unless trade-ready flags line up.",
        "stop": "AVWAPE (one level beyond the retested band), 2-close failure.",
        "targets": "50% at band 2, rest to band 3, trail band 1. Time stop 18 sessions.",
        "evidence": (
            "Worst measured mid-earnings variant at 10 sessions (-4.5%) in the 60d window — weight cut to 67. "
            "Demand the full confirmation stack before taking it."
        ),
    },
    "sma_breakout": {
        "label": "SMA Breakout + Retest",
        "group": "Main swing",
        "what": (
            "Reclaim of a major daily SMA (50/100/200) after time below it, then the retest/confirmation "
            "sequence: breakout date, pullback level, higher-high or previous-day-high confirmation."
        ),
        "detection": [
            "SMA_BREAKOUT_50/100/200_RECLAIM signal: close crosses the SMA after a sustained stretch on the "
            "other side (breakout date recorded).",
            "Retest tracked at the stored retest level; confirmation = higher high or previous-day high break "
            "(sma_breakout_confirmed with the trigger recorded).",
        ],
        "entry": "The confirmation bar after the retest — not the raw reclaim.",
        "stop": "The reclaimed SMA level (2-close failure).",
        "targets": "50% at band 2, rest to band 3, trail band 1. Time stop 18 sessions.",
        "evidence": (
            "SMA200 reclaims carried +0.90R edge in the playbook's correction window (mean-reversion regime) and "
            "faded to +0.20 in the bull window; 100/200 reclaims keep meaningful weights (92/108), 50 lighter (82)."
        ),
    },
    "top_pattern": {
        "label": "TOP Weekly Leader",
        "group": "Main swing",
        "what": (
            "Weekly-chart leaders: big 13w/26w gainers holding a shallow pullback from the 52w high, weekly "
            "EMA15 respected, above weekly SMA100 — then bought on specific daily re-entry triggers."
        ),
        "detection": [
            "Weekly screen: 13w/26w returns, pullback-from-52w-high %, weekly EMA15 hold ratio, weekly SMA100 test.",
            "Daily entry triggers (recorded per name): daily SMA50 reclaim or bounce, weekly SMA50 retest, or "
            "AVWAPE retest (top_pattern_entry_trigger + entry level).",
        ],
        "entry": "The recorded daily trigger level — the pattern is the context, the trigger is the trade.",
        "stop": "Beyond the trigger level (SMA50 or AVWAPE, 2-close failure).",
        "targets": "50% at band 2, rest to band 3, trail band 1; leaders often run past band 3 — trail rather than cap.",
        "evidence": (
            "Same regime as the weekly-8EMA strong basket: continuation entries (thrusts, power holds, quiet "
            "pullback resumes) are what works inside these names per the weekly-context backfill."
        ),
    },
    "favorite_zone_watch": {
        "label": "Favorite Zone Watch",
        "group": "Main swing",
        "what": (
            "Not yet a setup: the name sits in/near the favorite zone (AVWAPE to first dev) without today's "
            "trigger. It is on the board so the first real signal is not missed."
        ),
        "detection": ["Zone location only — no current-bar favorite signal."],
        "entry": "None yet. Wait for a bounce/cross/retest signal from the zone.",
        "stop": "n/a until a trigger defines it (then per the triggering family).",
        "targets": "n/a until triggered.",
        "evidence": "Watch bucket exists because zone location alone measured ~0 edge — the trigger is required.",
    },
    "general": {
        "label": "General / Untagged",
        "group": "Main swing",
        "what": "A tracked setup that did not map to a named family — usually mixed signals or legacy rows.",
        "detection": ["Fallback bucket; inspect the row's favorite signals to see what actually fired."],
        "entry": "Per the strongest current signal on the row.",
        "stop": "Protective band (LOWER_1 longs / UPPER_1 shorts), 2-close failure.",
        "targets": "50% at band 2, rest to band 3, trail band 1. Time stop 18 sessions.",
        "evidence": "",
    },
    # --- Study families (measured only; no scoring impact until proven) ---
    "weekly_ema8_hold_retest": {
        "label": "Weekly 8EMA Hold + Retest (study)",
        "group": "Study (measured only)",
        "what": (
            "Basket study: names whose last 10 weekly candles ALL closed at/above the weekly 8EMA — persistent "
            "institutional momentum. Rows record only on daily bounce entries inside the basket."
        ),
        "detection": [
            "Basket membership: 10 consecutive weekly closes >= weekly 8EMA (24+ weeks of history required).",
            "Study row records when the daily bar tags EMA15, EMA21, or the current-anchor first deviation and "
            "closes back above it.",
        ],
        "entry": "The daily tag-and-reclaim close inside the basket.",
        "stop": "The tagged level (EMA15/EMA21/UPPER_1), 2-close failure.",
        "targets": "50% at band 2, rest to band 3, trail band 1. Time stop 18 sessions.",
        "evidence": (
            "Weekly-context backfill: the weekly-strong regime is the biggest single lever found (baseline -0.10R "
            "vs -0.71R in mixed names). Forward study accruing in the tracker."
        ),
    },
    "htf_ema15_rejection": {
        "label": "HTF EMA15 Rejection (study)",
        "group": "Study (measured only)",
        "what": "1h/4h 15EMA pierce-and-close-back with both higher-timeframe trends aligned — an intraday-timeframe re-entry.",
        "detection": [
            "1h and 4h trend alignment required.",
            "The 1h/4h candle pierces its 15EMA and closes back on the trend side (session-aligned 4h resampling).",
        ],
        "entry": "On the rejection close (HTF timeframe).",
        "stop": "Beyond the rejecting 15EMA / rejection extreme.",
        "targets": "Per the daily family it feeds; study measures forward R.",
        "evidence": "Isolated study; not yet scored.",
    },
    "first_dev_breakout": {
        "label": "1st-Dev Breakout (study)",
        "group": "Study (measured only)",
        "what": "Close crosses the first deviation band — measured as its own family to compare against retest entries.",
        "detection": ["CROSS of UPPER_1/LOWER_1 with prior close inside the band."],
        "entry": "Signal close (study measures next-open fills too).",
        "stop": "Protective band, 2-close failure.",
        "targets": "50% at band 2, rest to band 3, trail band 1.",
        "evidence": "Playbook: +0.42/+0.14R edge across the two regime windows with tight stops (long side).",
    },
    "second_dev_breakout": {
        "label": "2nd-Dev Breakout (study)",
        "group": "Study (measured only)",
        "what": "Close crosses the second deviation band. Tracked as the control for chase entries.",
        "detection": ["CROSS of UPPER_2/LOWER_2 with prior close inside."],
        "entry": "Signal close (study).",
        "stop": "Protective band discipline.",
        "targets": "Study measurement only.",
        "evidence": (
            "Unstopped it is the second-worst long signal (-2.8% at 5 sessions); with tight stops it salvages a small "
            "positive edge. Deliberately scored near zero (weight 40) — the power HOLD is the tradable pattern, not the cross."
        ),
    },
    "playbook_volume_thrust": {
        "label": "Volume Thrust (playbook study)",
        "group": "Playbook research",
        "what": (
            "A >=1.5% move on >=2x 20-day volume, on the trend side of the anchored VWAP. The most regime-robust "
            "family in the playbook backfill — both sides."
        ),
        "detection": [
            "Close-to-close move >= +1.5% (longs; mirrored for shorts).",
            "Volume >= 2.0x the prior 20-session average.",
            "Close on the trend side of the current-anchor AVWAPE.",
        ],
        "entry": "Next session open after the thrust bar (as measured).",
        "stop": "0.1 ATR beyond the thrust bar's extreme held up in the backfill; level-based alternative is AVWAPE.",
        "targets": "Ride with the 18-session time stop; band 2/band 3 partials apply when bands are nearby.",
        "evidence": "+0.37/+0.49R edge (long) and +0.13/+0.33R (short) across both regime windows; t=3.4. Forward study live.",
    },
    "playbook_second_dev_power_hold": {
        "label": "2nd-Dev Power Hold (playbook study)",
        "group": "Playbook research",
        "what": (
            "The name LIVES beyond the second deviation band — 10+ sessions riding above UPPER_2. Continuation "
            "long only; the short mirror snaps back and is deliberately not recorded."
        ),
        "detection": [
            "Active second-stdev hold streak >= 10 sessions (the scan's mid-earnings zone streak).",
            "Close still beyond UPPER_2 today.",
        ],
        "entry": "Continuation: next open, or the first EMA8/first-dev tag-and-reclaim inside the hold.",
        "stop": "UPPER_1 (the zone floor), 2-close failure — a close back inside the first band ends the regime.",
        "targets": "Trail rather than target: these are the names that trend for weeks. Time stop still applies per episode.",
        "evidence": (
            "Zone streak >=10 was the strongest tracker factor found (+8.8% at 10 sessions, 89% win, n=45); playbook "
            "long edge +0.32/+0.21R in both windows, best inside weekly-strong names."
        ),
    },
    "playbook_quiet_pullback_resume": {
        "label": "Quiet Pullback Resume (playbook study)",
        "group": "Playbook research",
        "what": (
            "Three low-volume counter-trend sessions on the trend side of SMA50, then a resumption bar on rising "
            "volume. Robust both sides, both regime windows — the classic low-effort pullback."
        ),
        "detection": [
            "Close on the trend side of SMA50.",
            "Last 3 sessions each closed against the trend on below-average (20d) volume.",
            "Today closes with the trend on volume above yesterday's.",
        ],
        "entry": "Next open after the resumption bar.",
        "stop": "Under the pullback low (longs) or the SMA50 level, whichever is structurally closer; 2-close discipline on levels.",
        "targets": "50% at band 2 when bands are in range; otherwise ride to the 18-session time stop.",
        "evidence": "+0.24/+0.32R (long) and +0.22/+0.34R (short) edges across both windows. Forward study live.",
    },
}

# Legacy/alias family names that should resolve to a documented entry.
SETUP_DOC_ALIASES = {
    "avwap_breakdown": "avwap_breakout",
    "post_earnings_avwape_bounce": "post_earnings_avwap_bounce",
    "mid_earnings_first_dev_retest": "mid_earnings_1stdev_retest",
    "top_pattern_tracking": "top_pattern",
    "sma_breakout_tracking": "sma_breakout",
    # A completed/active hold beyond the 2nd stdev IS the power-hold regime.
    "mid_earnings_above_2nd_stdev": "playbook_second_dev_power_hold",
    # Bounce off the previous anchor's AVWAP behaves like a band bounce.
    "previous_avwape_bounce": "avwap_band_bounce",
}


def resolve_setup_doc(setup_family: str) -> tuple[str, dict]:
    """Return (canonical_key, doc) for a family name; falls back to 'general'.

    Accepts both machine keys ("mid_earnings_1stdev_retest") and the report's
    display labels ("mid earnings 1st-dev retest") — normalization lowercases,
    underscores spaces, and strips hyphens so label round-trips resolve.
    """
    key = str(setup_family or "").strip().lower().replace(" ", "_").replace("-", "")
    key = SETUP_DOC_ALIASES.get(key, key)
    if key in SETUP_DOCS:
        return key, SETUP_DOCS[key]
    return "general", SETUP_DOCS["general"]


def resolve_setup_family_from_candidates(candidates) -> str:
    """First candidate (family key, display label, or tag) that resolves to a
    real documented family; 'general' only when nothing does."""
    for candidate in candidates or []:
        key, _doc = resolve_setup_doc(candidate)
        if key != "general":
            return key
    return "general"


def all_setup_docs_by_group() -> list[tuple[str, list[tuple[str, dict]]]]:
    """Docs grouped for display, preserving a sensible reading order."""
    group_order = ["Main swing", "Earnings cycle", "Study (measured only)", "Playbook research"]
    grouped: dict[str, list[tuple[str, dict]]] = {name: [] for name in group_order}
    for key, doc in SETUP_DOCS.items():
        grouped.setdefault(doc.get("group", "Main swing"), []).append((key, doc))
    return [(name, grouped.get(name, [])) for name in group_order if grouped.get(name)]


# ---------------------------------------------------------------------------
# Concrete trade plan from live band levels
# ---------------------------------------------------------------------------
def _to_float(value) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_trade_plan(
    *,
    side: str,
    setup_family: str = "",
    favorite_signals: tuple | list = (),
    bands: dict | None = None,
    vwap=None,
    atr20=None,
    last_close=None,
) -> dict:
    """Turn the family's stop/target rules into concrete prices.

    ``bands``/``vwap`` come from the scan's current earnings anchor (ai_state).
    Returns labels, prices, risk per share and R multiples where computable;
    missing inputs degrade to labels-only guidance instead of failing.
    """
    side = _norm_side(side)
    bands = bands if isinstance(bands, dict) else {}
    vwap_value = _to_float(vwap)
    close_value = _to_float(last_close)
    family_key, _doc = resolve_setup_doc(setup_family)

    signals = {str(s or "").strip().upper() for s in (favorite_signals or [])}
    is_post_earnings = family_key.startswith("post_earnings")
    if _FIRST_BAND_BOUNCE_SIGNALS[side] in signals:
        stop_label = "AVWAPE"
        stop_price = vwap_value
        stop_reason = "first-band bounce: stop one level beyond the bounced band (the anchored VWAP)"
    elif family_key == "mid_earnings_ema15_retest":
        stop_label, stop_price, stop_reason = "EMA_15", None, "retested EMA is the stop level"
    elif family_key == "mid_earnings_ema21_retest":
        stop_label, stop_price, stop_reason = "EMA_21", None, "retested EMA is the stop level"
    elif family_key == "playbook_second_dev_power_hold":
        stop_label = favorable_band_label(side, 1)
        stop_price = _to_float(bands.get(stop_label))
        stop_reason = "zone floor: a close back inside the first band ends the power-hold regime"
    elif is_post_earnings:
        stop_label = "POST_EARNINGS_AVWAPE"
        stop_price = None
        stop_reason = "post-earnings anchor VWAP; 1-close failure discipline"
    else:
        stop_label = protective_band_label(side)
        stop_price = _to_float(bands.get(stop_label))
        stop_reason = "protective band one level beyond the favorite zone"

    partial_label = favorable_band_label(side, 2)
    final_label = favorable_band_label(side, 3)
    partial_price = _to_float(bands.get(partial_label))
    final_price = _to_float(bands.get(final_label))

    risk = None
    if close_value is not None and stop_price is not None:
        risk = (close_value - stop_price) if side == "LONG" else (stop_price - close_value)
        if risk <= 0:
            risk = None  # price is already beyond the stop level -> plan is stale

    def _r_multiple(target) -> float | None:
        if risk is None or target is None or close_value is None:
            return None
        reward = (target - close_value) if side == "LONG" else (close_value - target)
        return reward / risk if risk > 0 else None

    return {
        "side": side,
        "setup_family": family_key,
        "entry_reference": close_value,
        "stop_label": stop_label,
        "stop_price": stop_price,
        "stop_reason": stop_reason,
        "stop_close_failures": POST_EARNINGS_STOP_CLOSE_FAILURES if is_post_earnings else STOP_CLOSE_FAILURES,
        "risk_per_share": risk,
        "risk_pct_of_price": (risk / close_value * 100.0) if risk and close_value else None,
        "partial_label": partial_label,
        "partial_price": partial_price,
        "partial_r": _r_multiple(partial_price),
        "final_label": final_label,
        "final_price": final_price,
        "final_r": _r_multiple(final_price),
        "trail_label": favorable_band_label(side, 1),
        "time_stop_sessions": TIME_STOP_SESSIONS,
        "atr20": _to_float(atr20),
        "plan_summary": (
            f"Stop: {stop_label} ({stop_reason}). Take 50% at {partial_label}, run the rest to "
            f"{final_label} with the stop trailed to {favorable_band_label(side, 1)} after the partial. "
            f"Time stop {TIME_STOP_SESSIONS} sessions."
        ),
    }
