"""Recipe simulation and the ``house_default_v1`` outcome contract (Phase 6).

An R number means nothing without the policy that produced it. Every row here
names its ``recipe_id`` (entry, stop, management, time stop) and its
``outcome_definition_id`` (fills, costs, ambiguity, maturity), and alternative
recipes on one occurrence are correlated diagnostics of **one** episode - never
extra samples.

``house_default_v1`` (sec 14.2), implemented exactly:

* ``net_r = gross_r - (2 x (commission_per_share + half_spread) + slippage)
  / stop_distance_$`` where slippage is +1 half_spread on stop/market entries -
  every slice recipe enters via a declared market-type order (the precommitted
  MOC or the completed-bar close), so every row pays it;
* commission $0.0035/share (IBKR tiered);
* ``half_spread`` = observed NBBO at signal when supplied, else the declared
  fallback ``max($0.01, 2bp x price)``;
* same-bar ambiguity: **STOP_FIRST is primary**, and the TARGET_FIRST reading
  is retained as ``r_upper_bound`` with ``path_resolution = AMBIGUOUS``;
* maturity: ``min(EOD, stop)`` intraday; ``min(+18 sessions,
  stop/target/expiry)`` swing - a resolved trade matures when it resolves.

Rules the simulators hold to:

* **The walk never leaves its recipe's window.** Swing paths stop at the
  18-session time stop; intraday paths stop at the entry session's close.
  Bars beyond the window are not this recipe's outcome.
* **An unresolved path carries no realized R.** ``OPEN`` and ``TRUNCATED``
  rows keep their checkpoints and MFE/MAE (path facts) but ``gross_r`` and
  ``net_r`` stay null - an interim reading must never enter a mean.
* **Non-terminal rows are recomputed and superseded.** ``build_outcomes``
  re-simulates ``OPEN``/``TRUNCATED``/``NO_TRIGGER`` rows as more path
  arrives and publishes a superseding row (append-only; readers take the
  latest ``computed_at`` per (occurrence, recipe, definition) via
  :func:`latest_outcomes`). Terminal rows are immutable evidence and are
  never recomputed.

``MATURED`` is deliberately not a result state: it is the derived predicate
``maturity_at <= as_of`` (:func:`is_matured`). Storing it would freeze a
question whose answer changes with the asking time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

try:  # package import
    from . import exchange_calendar as xcal
    from .manifest import utc_now
    from .schemas import RESULT_STATES, SCHEMA_VERSION
    from .store import ResearchStore
except ImportError:  # pragma: no cover - scripts/ directly on sys.path
    import exchange_calendar as xcal  # type: ignore
    from manifest import utc_now  # type: ignore
    from schemas import RESULT_STATES, SCHEMA_VERSION  # type: ignore
    from store import ResearchStore  # type: ignore

OUTCOME_DEFINITION_ID = "house_default_v1"
COMMISSION_PER_SHARE = 0.0035
MIN_HALF_SPREAD = 0.01
HALF_SPREAD_BPS = 0.0002  # 2bp of price
#: sec 14.2: +1 half_spread of slippage on stop/market entries. Every slice
#: recipe enters via a declared market-type order, so the simulators pass this.
ENTRY_SLIPPAGE_HALF_SPREADS = 1.0
SWING_TIME_STOP_SESSIONS = 18

ANALYSIS_UNIT_OPPORTUNITY = "OPPORTUNITY"
ANALYSIS_UNIT_ATTEMPT = "ATTEMPT"

PATH_EXACT = "EXACT"
PATH_LOWER_TIMEFRAME = "LOWER_TIMEFRAME"
PATH_AMBIGUOUS = "AMBIGUOUS"

STATE_NO_TRIGGER = "NO_TRIGGER"
STATE_OPEN = "OPEN"
STATE_STOPPED = "STOPPED"
STATE_TARGETED = "TARGETED"
STATE_EXPIRED = "EXPIRED"
STATE_TRUNCATED = "TRUNCATED"
STATE_CENSORED = "CENSORED"
STATE_AMBIGUOUS_BAR = "AMBIGUOUS_BAR"

#: States whose row is final evidence. OPEN/TRUNCATED (path not finished or
#: not yet archived) and NO_TRIGGER (the thesis may still trigger) are
#: recomputed on later builds and superseded by ``computed_at``.
TERMINAL_RESULT_STATES = frozenset(
    {STATE_STOPPED, STATE_TARGETED, STATE_EXPIRED, STATE_AMBIGUOUS_BAR, STATE_CENSORED}
)

SWING_CHECKPOINTS = (("r_at_s1", 1), ("r_at_s2", 2), ("r_at_s3", 3), ("r_at_s5", 5), ("r_at_s10", 10), ("r_at_s18", 18))
INTRADAY_CHECKPOINTS = (("r_at_15m", 15), ("r_at_30m", 30), ("r_at_60m", 60), ("r_at_120m", 120))


@dataclass(frozen=True)
class Recipe:
    """One declared trade policy. Named on every row it produces."""

    recipe_id: str
    timeframe: str  # D1 | M5
    analysis_unit: str
    entry: str
    stop: str
    management: str
    time_stop_sessions: int | None = None
    time_stop_minutes: int | None = None
    #: Consecutive closes beyond the stop level required to invalidate.
    close_failures: int = 1
    target_r: float | None = None
    is_control: bool = False
    is_diagnostic: bool = False
    #: For the approved next-session M5-close study.  A tracker source name
    #: selects one recorded structural candidate; an ATR multiple is a bounded
    #: control.  Neither field changes a champion detector.
    stop_selector: str = ""
    stop_atr_multiple: float | None = None
    #: Phase 0.12 B3, higher-timeframe LRSI entry study. Declarative, so the
    #: recipe id and the rule that produced a row can never drift apart.
    htf_timeframe: str = ""
    cross_level: float | None = None
    cross_direction: str = ""  # "up" | "down"
    note: str = ""


SWING_HOUSE_V1 = Recipe(
    recipe_id="swing_house_v1",
    timeframe="D1",
    analysis_unit=ANALYSIS_UNIT_OPPORTUNITY,
    # Sec 12.1 forbids *assuming* a same-close fill; this recipe declares one
    # explicitly as a precommitted market-on-close order, which is the
    # exception that section allows. signal_known_at == entry_eligible_at.
    entry="signal_close_precommitted_moc",
    stop="structural_level_close_failure",
    management="partial_at_band2_trail_band1_run_band3",
    time_stop_sessions=SWING_TIME_STOP_SESSIONS,
    close_failures=2,
    note="1 close for post-earnings families (see POST_EARNINGS_CLOSE_FAILURES)",
)

INTRADAY_BOUNCE_V1 = Recipe(
    recipe_id="intraday_bounce_v1",
    timeframe="M5",
    analysis_unit=ANALYSIS_UNIT_ATTEMPT,
    entry="completed_bounce_bar_close",
    stop="production_per_bounce_type",
    management="quick_r_60m_then_eod",
    time_stop_minutes=None,
    close_failures=1,
)

CONTROL_FIXED_1R2R_V1 = Recipe(
    recipe_id="control_fixed_1r2r_v1",
    timeframe="D1",
    analysis_unit=ANALYSIS_UNIT_OPPORTUNITY,
    entry="signal_close_precommitted_moc",
    stop="fixed_1r",
    management="fixed_2r_target",
    time_stop_sessions=SWING_TIME_STOP_SESSIONS,
    target_r=2.0,
    is_control=True,
)

CONTROL_TIME_ONLY_V1 = Recipe(
    recipe_id="control_time_only_v1",
    timeframe="D1",
    analysis_unit=ANALYSIS_UNIT_OPPORTUNITY,
    entry="signal_close_precommitted_moc",
    stop="none",
    management="time_only_exit",
    time_stop_sessions=SWING_TIME_STOP_SESSIONS,
    is_control=True,
)

DIAGNOSTIC_ATR_STOP_V1 = Recipe(
    recipe_id="diag_signal_bar_atr_stop_v1",
    timeframe="M5",
    analysis_unit=ANALYSIS_UNIT_ATTEMPT,
    entry="completed_bounce_bar_close",
    stop="signal_bar_extreme_plus_0_25_atr_m5_14",
    management="quick_r_60m_then_eod",
    is_diagnostic=True,
    note="registered diagnostic, never the primary - tracker parity carries over unchanged",
)


M5_CLOSE_STOP_SOURCES = (
    "current_anchor",
    "sma",
    "ema",
    "post_earnings_anchor",
    "post_earnings_candle",
)
M5_CLOSE_STOP_RANKS = (1, 2, 3)
M5_CLOSE_TARGETS_R = (1.0, 2.0, 3.0)
M5_CLOSE_ATR_MULTIPLES = (0.5, 1.0, 1.5)


def _m5_close_recipes() -> tuple[Recipe, ...]:
    """The bounded registered grid; never a free Cartesian search."""
    recipes: list[Recipe] = []
    for source in M5_CLOSE_STOP_SOURCES:
        for rank in M5_CLOSE_STOP_RANKS:
            for target in M5_CLOSE_TARGETS_R:
                recipes.append(
                    Recipe(
                        recipe_id=f"m5close_{source}{rank}_{target:g}r_v1",
                        timeframe="M5_OPPORTUNITY",
                        analysis_unit=ANALYSIS_UNIT_OPPORTUNITY,
                        entry="next_session_first_completed_m5_close",
                        stop=f"tracker_{source}_nearest_{rank}_close_failure",
                        management=f"fixed_{target:g}r_target",
                        time_stop_sessions=SWING_TIME_STOP_SESSIONS,
                        target_r=target,
                        stop_selector=f"{source}:{rank}",
                        note="ranked structural level and close count preserved from the tracker scenario",
                    )
                )
    for multiple in M5_CLOSE_ATR_MULTIPLES:
        for target in M5_CLOSE_TARGETS_R:
            recipes.append(
                Recipe(
                    recipe_id=f"m5close_atr{multiple:g}_{target:g}r_v1",
                    timeframe="M5_OPPORTUNITY",
                    analysis_unit=ANALYSIS_UNIT_OPPORTUNITY,
                    entry="next_session_first_completed_m5_close",
                    stop=f"entry_minus_{multiple:g}_atr_m5_14",
                    management=f"fixed_{target:g}r_target",
                    time_stop_sessions=SWING_TIME_STOP_SESSIONS,
                    target_r=target,
                    stop_atr_multiple=multiple,
                    close_failures=1,
                    is_control=True,
                    note="ATR placement control; hard intrabar stop; no M1 or bid/ask input",
                )
            )
    return tuple(recipes)


M5_CLOSE_RECIPES = _m5_close_recipes()


# ---------------------------------------------------------------------------
# Phase 0.12 B3: higher-timeframe LRSI entry study. SHADOW ONLY.
#
# The question is the trader's, asked 2026-09-01: is there anything in entering
# a Focus-style setup on an LRSI cross at M30/H1/H2/H4 rather than on M5? This
# lane answers it with outcome rows and nothing else. It reaches no detector,
# no score, no alert, no Focus list and no review queue, and promotion is
# plan.md sec 7's job, not this module's.
#
# **Bounded, never a Cartesian search.** Four timeframes x four entries is the
# whole grid - 16 registered recipes, each with ONE stop model and ONE target,
# following the `DIAGNOSTIC_ATR_STOP_V1` precedent. Every one carries
# `is_diagnostic=True`.
#
# **Alternative recipes on one occurrence are correlated diagnostics of ONE
# episode, never extra samples.** That is the module's standing rule and it
# binds hardest here: sixteen readings of one setup are sixteen views of one
# trade, and averaging them as sixteen trades would manufacture confidence out
# of nothing.
#
# **The long and short legs read the SAME series.** The efficiency formula
# clamps at zero, so an unmirrored down-cross measures the UP move's efficiency
# collapsing rather than a down move's strength - a different feature from the
# live M5 engines' mirrored-close idiom, and deliberately so. The decision, its
# cost and its fixture are in `indicators/efficiency_lrsi.RESEARCH_CROSS_LEVELS`.
#
# **Stubs are excluded from the LRSI input.** RTH is 6.5 hours, so H2 and H4
# both end each session with a short bucket. An H4 "bar" covering 13:30-16:00
# is not an H4 bar; feeding it to an EMA would make the oscillator measure a
# duration that changes with the time of day. Completed bars only, and a stub
# is a completed bucket of the wrong length.
HTF_LRSI_TIMEFRAMES = ("M30", "H1", "H2", "H4")

#: (side, direction, level). Longs read the crossing UP through the two levels
#: the trader already reads; shorts read the crossing DOWN through 50 and 80.
HTF_LRSI_ENTRIES = (
    ("LONG", "up", 50.0),
    ("LONG", "up", 20.0),
    ("SHORT", "down", 50.0),
    ("SHORT", "down", 80.0),
)

#: One target, not the three the M5-close grid carries. The prompt authorizing
#: this named SIXTEEN recipes, and 4 x 4 x 3 is forty-eight; sixteen is also
#: what keeps the nightly inside `setup_research`'s reserve. 2.0R is the middle
#: of `M5_CLOSE_TARGETS_R` and the same target the fixed-R control already
#: uses, so the two are directly comparable. Widening to the full set is this
#: one constant.
HTF_LRSI_TARGETS_R = (2.0,)

#: The single stop model: the signal bar's own extreme, pushed out a quarter of
#: an ATR, exactly as `DIAGNOSTIC_ATR_STOP_V1` does on M5. ATR is measured on
#: the SAME timeframe as the entry - an M5 ATR under an H4 entry would size the
#: risk off a bar the recipe never looks at.
HTF_LRSI_STOP_ATR_MULTIPLE = 0.25


def _htf_lrsi_recipes() -> tuple[Recipe, ...]:
    """The bounded registered grid; never a free Cartesian search."""
    recipes: list[Recipe] = []
    for timeframe in HTF_LRSI_TIMEFRAMES:
        for side, direction, level in HTF_LRSI_ENTRIES:
            for target in HTF_LRSI_TARGETS_R:
                recipes.append(
                    Recipe(
                        recipe_id=(
                            f"htf_lrsi_{timeframe.lower()}_{direction}{level:g}"
                            f"_{target:g}r_v1"
                        ),
                        timeframe="HTF_LRSI",
                        analysis_unit=ANALYSIS_UNIT_OPPORTUNITY,
                        entry=(
                            f"{timeframe.lower()}_efficiency_lrsi_cross_"
                            f"{direction}_{level:g}_bar_close"
                        ),
                        stop=(
                            f"signal_bar_extreme_plus_"
                            f"{HTF_LRSI_STOP_ATR_MULTIPLE:g}_atr_{timeframe.lower()}_14"
                        ),
                        management=f"fixed_{target:g}r_target",
                        time_stop_sessions=SWING_TIME_STOP_SESSIONS,
                        target_r=target,
                        stop_atr_multiple=HTF_LRSI_STOP_ATR_MULTIPLE,
                        htf_timeframe=timeframe,
                        cross_level=level,
                        cross_direction=direction,
                        is_diagnostic=True,
                        note=(
                            f"shadow-only HTF LRSI study; {side} leg; derived "
                            "bars exclude session stubs"
                        ),
                    )
                )
    return tuple(recipes)


HTF_LRSI_RECIPES = _htf_lrsi_recipes()

RECIPES = {
    recipe.recipe_id: recipe
    for recipe in (
        SWING_HOUSE_V1,
        INTRADAY_BOUNCE_V1,
        CONTROL_FIXED_1R2R_V1,
        CONTROL_TIME_ONLY_V1,
        DIAGNOSTIC_ATR_STOP_V1,
    )
}

#: Post-earnings families invalidate on a single close (sec 19.3).
POST_EARNINGS_CLOSE_FAILURES = {"POST_EARNINGS_CANDLE_BREAK": 1}

#: swing_house_v1 is the primary recipe for both slice setups; both are D1
#: swing families reported by the master-scan detector (sec 19.3, normative).
PRIMARY_RECIPE_BY_SETUP = {
    "AVWAPE_TO_FIRST_DEV": SWING_HOUSE_V1.recipe_id,
    "POST_EARNINGS_CANDLE_BREAK": SWING_HOUSE_V1.recipe_id,
}


def half_spread(price: float, observed_nbbo_half_spread: float | None = None) -> float:
    """Observed NBBO half-spread at signal, else the declared fallback."""
    if observed_nbbo_half_spread is not None:
        try:
            return max(0.0, float(observed_nbbo_half_spread))
        except (TypeError, ValueError):
            pass
    return max(MIN_HALF_SPREAD, HALF_SPREAD_BPS * float(price))


def net_r(
    gross_r: float,
    stop_distance: float,
    price: float,
    *,
    observed_half_spread=None,
    entry_slippage_half_spreads: float = 0.0,
) -> float:
    """``gross_r - (2 x (commission + half_spread) + slippage) / stop_distance``.

    The 2x term is sec 14.2's round-trip formula verbatim;
    ``entry_slippage_half_spreads`` carries its separate "+1 half_spread
    slippage on stop/market entries" bullet. The simulators pass
    :data:`ENTRY_SLIPPAGE_HALF_SPREADS` because every slice recipe enters via
    a declared market-type order; a future limit-entry recipe would pass 0.
    """
    if not stop_distance:
        return gross_r
    spread = half_spread(price, observed_half_spread)
    cost = (2.0 * (COMMISSION_PER_SHARE + spread) + float(entry_slippage_half_spreads) * spread) / abs(
        float(stop_distance)
    )
    return float(gross_r) - cost


def is_matured(row, as_of: datetime) -> bool:
    """MATURED is derived, never stored (sec 14.2)."""
    maturity = row.get("maturity_at") if isinstance(row, dict) else row
    if maturity is None:
        return False
    if maturity.tzinfo is None:
        maturity = maturity.replace(tzinfo=timezone.utc)
    return maturity <= as_of


def _r(price, entry: float, stop_distance: float, side: str) -> float | None:
    if price is None or not stop_distance:
        return None
    direction = 1.0 if str(side).upper() == "LONG" else -1.0
    return direction * (float(price) - float(entry)) / float(stop_distance)


@dataclass
class OutcomeReport:
    dataset: str = "outcome_path"
    status: str = "OK"  # OK | DISABLED | NOTHING_TO_SIMULATE
    rows: int = 0
    by_recipe: dict = field(default_factory=dict)
    skipped: dict = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


@dataclass
class _Walk:
    """What one pass over the recipe's own bar window concluded."""

    result_state: str = STATE_OPEN
    first_hit: str | None = None
    first_hit_at: datetime | None = None
    #: When the position finished (last piece exited). Feeds maturity's min().
    resolved_at: datetime | None = None
    path_resolution: str = PATH_EXACT
    r_lower: float | None = None
    r_upper: float | None = None
    #: Realized R. Stays None while the path is unresolved (OPEN/TRUNCATED).
    gross: float | None = None
    mfe: float = 0.0
    mae: float = 0.0
    time_to_mfe: int | None = None

    def track(self, bar, entry_price, stop_distance, side, offset) -> None:
        high_r = _r(bar.get("high"), entry_price, stop_distance, side)
        low_r = _r(bar.get("low"), entry_price, stop_distance, side)
        if high_r is None or low_r is None:
            return
        favourable = max(high_r, low_r)
        adverse = min(high_r, low_r)
        if favourable > self.mfe:
            self.mfe = favourable
            self.time_to_mfe = offset * 24 * 60
        if adverse < self.mae:
            self.mae = adverse


def _walk_plain(
    horizon,
    *,
    entry_price: float,
    stop_distance: float,
    side: str,
    stop_price,
    has_stop: bool,
    close_failures: int,
    target_price,
    time_stop_sessions: int | None,
) -> _Walk:
    """The plain stop/target walk, bounded to ``horizon``.

    Same-bar ambiguity keeps the LD-07/BD-40 doctrine exactly: when one
    session's range contains both the stop and the target, STOP_FIRST is the
    primary and the TARGET_FIRST reading is retained as the upper bound.
    """
    walk = _Walk()
    consecutive = 0
    for offset, bar in enumerate(horizon, start=1):
        walk.track(bar, entry_price, stop_distance, side, offset)
        target_hit = target_price is not None and _reached(bar, target_price, side, favourable=True)
        stop_hit = False
        if has_stop and stop_price is not None:
            close_beyond = _beyond_stop(bar.get("close"), float(stop_price), side)
            consecutive = consecutive + 1 if close_beyond else 0
            stop_hit = consecutive >= close_failures

        if target_hit and stop_hit:
            walk.result_state = STATE_AMBIGUOUS_BAR
            walk.path_resolution = PATH_AMBIGUOUS
            walk.first_hit = "STOP"
            walk.first_hit_at = walk.resolved_at = _session_close(bar)
            walk.r_lower = _r(stop_price, entry_price, stop_distance, side)
            walk.r_upper = _r(target_price, entry_price, stop_distance, side)
            walk.gross = walk.r_lower
            return walk
        if stop_hit:
            walk.result_state = STATE_STOPPED
            walk.first_hit = "STOP"
            walk.first_hit_at = walk.resolved_at = _session_close(bar)
            walk.gross = _r(bar.get("close"), entry_price, stop_distance, side)
            return walk
        if target_hit:
            walk.result_state = STATE_TARGETED
            walk.first_hit = "TARGET"
            walk.first_hit_at = walk.resolved_at = _session_close(bar)
            walk.gross = _r(target_price, entry_price, stop_distance, side)
            return walk

    if time_stop_sessions is not None and len(horizon) >= time_stop_sessions:
        walk.result_state = STATE_EXPIRED
        walk.first_hit = "NEITHER"
        walk.resolved_at = _session_close(horizon[-1])
        walk.gross = _r(horizon[-1].get("close"), entry_price, stop_distance, side)
    # else: OPEN - the path is simply not complete yet, and an unresolved
    # path carries no realized R.
    return walk


def _walk_managed(
    horizon,
    *,
    entry_price: float,
    stop_distance: float,
    side: str,
    stop_price,
    close_failures: int,
    band_1: float,
    band_2: float,
    band_3,
    time_stop_sessions: int | None,
) -> _Walk:
    """``partial_at_band2_trail_band1_run_band3``, executed over ``horizon``.

    Per bar, in order: intra-bar favourable fills (the 50% partial at band 2,
    the runner exit at band 3 - band 3 implies band 2 was crossed first, so
    the same bar can do both), then the close events (the structural
    close-failure stop, and after the partial, the band-1 trail). The stop and
    the trail apply to whatever is still on - a stop after the partial exits
    the remaining half at that close, and the partial stays credited.

    Same-bar conservatism (the LD-07/BD-40 doctrine): when a bar offers a NEW
    favourable fill, ends the position on its close, AND touched the stop
    level intra-bar, the fill-vs-stop ordering is unknowable from OHLC - the
    row is AMBIGUOUS_BAR, the primary reading takes the exit without crediting
    that bar's fills, and the fill-credited reading is retained as
    ``r_upper_bound``. A pure band-1 trail with no stop touch is NOT
    ambiguous: an intra-bar band fill definitionally precedes the bar's close,
    so the partial is credited and the runner exits at that close.
    """
    walk = _Walk()
    consecutive = 0
    partial_taken = False
    partial_r: float | None = None

    def _blend(runner_r):
        if runner_r is None:
            return None
        if partial_taken and partial_r is not None:
            return 0.5 * partial_r + 0.5 * float(runner_r)
        return float(runner_r)

    for offset, bar in enumerate(horizon, start=1):
        walk.track(bar, entry_price, stop_distance, side, offset)

        fills: list[tuple[str, float]] = []
        if not partial_taken and _reached(bar, band_2, side, favourable=True):
            fills.append(("partial", band_2))
        runner_target_hit = (
            band_3 is not None
            and (partial_taken or fills)
            and _reached(bar, band_3, side, favourable=True)
        )
        if runner_target_hit:
            fills.append(("runner", band_3))

        close = bar.get("close")
        close_beyond_stop = stop_price is not None and _beyond_stop(close, float(stop_price), side)
        consecutive = consecutive + 1 if close_beyond_stop else 0
        stop_completes = consecutive >= close_failures
        partial_this_bar = any(kind == "partial" for kind, _level in fills)
        trail_exits = (partial_taken or partial_this_bar) and _beyond_stop(close, band_1, side)
        close_exit = stop_completes or trail_exits
        stop_touched = stop_price is not None and _reached(bar, float(stop_price), side, favourable=False)

        if fills and close_exit and stop_touched:
            # New favourable fill and a position-ending close in one bar:
            # conservative primary takes the exit without this bar's fills.
            walk.result_state = STATE_AMBIGUOUS_BAR
            walk.path_resolution = PATH_AMBIGUOUS
            walk.first_hit = "STOP"
            walk.first_hit_at = walk.resolved_at = _session_close(bar)
            walk.gross = walk.r_lower = _blend(_r(close, entry_price, stop_distance, side))
            optimistic_partial = partial_r if partial_taken else _r(band_2, entry_price, stop_distance, side)
            optimistic_runner = (
                _r(band_3, entry_price, stop_distance, side)
                if runner_target_hit
                else _r(close, entry_price, stop_distance, side)
            )
            if optimistic_partial is not None and optimistic_runner is not None:
                walk.r_upper = 0.5 * optimistic_partial + 0.5 * optimistic_runner
            return walk

        for kind, level in fills:
            if kind == "partial":
                partial_taken = True
                partial_r = _r(level, entry_price, stop_distance, side)
                if walk.first_hit is None:
                    walk.first_hit = "TARGET"
                    walk.first_hit_at = _session_close(bar)
            else:  # the runner reached band 3: the position is done
                walk.result_state = STATE_TARGETED
                if walk.first_hit is None:
                    walk.first_hit = "TARGET"
                    walk.first_hit_at = _session_close(bar)
                walk.resolved_at = _session_close(bar)
                walk.gross = _blend(_r(level, entry_price, stop_distance, side))
                return walk

        if close_exit:
            # Stop or trail: whatever is still on exits at this close.
            walk.result_state = STATE_STOPPED
            if walk.first_hit is None:
                walk.first_hit = "STOP"
                walk.first_hit_at = _session_close(bar)
            walk.resolved_at = _session_close(bar)
            walk.gross = _blend(_r(close, entry_price, stop_distance, side))
            return walk

    if time_stop_sessions is not None and len(horizon) >= time_stop_sessions:
        walk.result_state = STATE_EXPIRED
        if walk.first_hit is None:
            walk.first_hit = "NEITHER"
        walk.resolved_at = _session_close(horizon[-1])
        walk.gross = _blend(_r(horizon[-1].get("close"), entry_price, stop_distance, side))
    return walk


def _band(bands, side: str, number: int):
    key = f"UPPER_{number}" if str(side).upper() == "LONG" else f"LOWER_{number}"
    value = (bands or {}).get(key)
    return None if value is None else float(value)


def simulate_swing(
    occurrence: dict,
    bars,
    recipe: Recipe,
    *,
    bands=None,
    as_of: datetime,
    computed_at: datetime | None = None,
    observed_half_spread: float | None = None,
    intraday_bars=None,
    run_id: str = "",
) -> dict | None:
    """Simulate one D1 recipe over the sessions after the trigger.

    ``bars`` are canonical D1 rows sorted ascending, including the trigger
    session. The walk is bounded by the recipe's own time stop; bars past it
    are not this recipe's outcome. ``bands`` must be the AVWAP band levels as
    of the signal session - a later band set would be look-ahead.
    """
    stamp = computed_at or utc_now()
    side = str(occurrence.get("side") or "LONG").upper()
    trigger_at = occurrence.get("trigger_at")
    if trigger_at is None:
        return _no_trigger_row(occurrence, recipe, stamp, run_id)

    ordered = sorted(
        [row for row in bars if row.get("session_date") is not None],
        key=lambda row: _as_date(row["session_date"]),
    )
    trigger_date = trigger_at.date() if isinstance(trigger_at, datetime) else _as_date(trigger_at)
    entry_index = next(
        (index for index, row in enumerate(ordered) if _as_date(row["session_date"]) == trigger_date), None
    )
    if entry_index is None:
        return None

    entry_bar = ordered[entry_index]
    entry_price = float(occurrence.get("entry_price_ref") or entry_bar.get("close"))
    stop_price = occurrence.get("stop_price_ref")
    forward = ordered[entry_index + 1 :]
    stop_distance = abs(entry_price - float(stop_price)) if stop_price is not None else None
    if not stop_distance:
        return None

    close_failures = POST_EARNINGS_CLOSE_FAILURES.get(
        str(occurrence.get("canonical_setup_id") or ""), recipe.close_failures
    )
    direction = 1.0 if side == "LONG" else -1.0

    session = xcal.trading_session(trigger_date)
    entry_at = session.rth_close_at if session else trigger_at
    maturity_projected, _expiry_index = _swing_maturity(ordered, entry_index, recipe)

    # The time stop bounds the path we walk: sessions past it are not part of
    # this recipe's outcome at all - not for stops, targets, or management.
    horizon = forward[: recipe.time_stop_sessions] if recipe.time_stop_sessions else forward

    band_1 = _band(bands, side, 1)
    band_2 = _band(bands, side, 2)
    band_3 = _band(bands, side, 3)
    managed = (
        recipe.management.startswith("partial_at_band2") and band_1 is not None and band_2 is not None
    )
    if managed:
        walk = _walk_managed(
            horizon,
            entry_price=entry_price,
            stop_distance=stop_distance,
            side=side,
            stop_price=stop_price,
            close_failures=close_failures,
            band_1=band_1,
            band_2=band_2,
            band_3=band_3,
            time_stop_sessions=recipe.time_stop_sessions,
        )
    else:
        target_price = None
        if recipe.target_r is not None:
            target_price = entry_price + direction * recipe.target_r * stop_distance
        elif recipe.management.startswith("partial_at_band2") and band_3 is not None:
            # Bands too incomplete to manage: fall back to the plain path with
            # band 3 as the target (BD-42's declared fallback).
            target_price = band_3
        walk = _walk_plain(
            horizon,
            entry_price=entry_price,
            stop_distance=stop_distance,
            side=side,
            stop_price=stop_price,
            has_stop=recipe.stop != "none",
            close_failures=close_failures,
            target_price=target_price,
            time_stop_sessions=recipe.time_stop_sessions,
        )

    result_state = walk.result_state
    if result_state == STATE_OPEN and is_matured({"maturity_at": maturity_projected}, as_of):
        # The clock says this should have resolved, but the bars ran out:
        # truncated evidence, never quietly reported as a finished trade.
        result_state = STATE_TRUNCATED

    # sec 14.2: swing maturity is min(+18 sessions, stop/target/expiry) - a
    # resolved trade matures when it resolves, not weeks later.
    maturity_at = walk.resolved_at if walk.resolved_at is not None else maturity_projected

    gross = walk.gross
    row = {
        "occurrence_id": occurrence.get("occurrence_id"),
        "recipe_id": recipe.recipe_id,
        "outcome_definition_id": OUTCOME_DEFINITION_ID,
        "analysis_unit": recipe.analysis_unit,
        "entry_at": entry_at,
        "entry_price": entry_price,
        "stop_price": _number(stop_price),
        "stop_distance": stop_distance,
        "mfe_r": walk.mfe,
        "mae_r": walk.mae,
        "time_to_mfe_min": walk.time_to_mfe,
        "first_hit": walk.first_hit,
        "first_hit_at": walk.first_hit_at,
        "path_resolution": walk.path_resolution,
        "r_lower_bound": walk.r_lower,
        "r_upper_bound": walk.r_upper,
        "gross_r": gross,
        "net_r": None
        if gross is None
        else net_r(
            gross,
            stop_distance,
            entry_price,
            observed_half_spread=observed_half_spread,
            entry_slippage_half_spreads=ENTRY_SLIPPAGE_HALF_SPREADS,
        ),
        "cost_model_id": OUTCOME_DEFINITION_ID,
        "result_state": result_state,
        "maturity_at": maturity_at,
        "censor_reason": None,
        "computed_at": stamp,
        "input_capture_mode_worst": _worst_capture_mode(row.get("capture_mode") for row in ordered),
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
    }
    for column, sessions in SWING_CHECKPOINTS:
        row[column] = (
            _r(forward[sessions - 1].get("close"), entry_price, stop_distance, side)
            if len(forward) >= sessions
            else None
        )
    for column, _minutes in INTRADAY_CHECKPOINTS:
        row[column] = None
    row["r_at_eod"] = None
    if intraday_bars:
        _fill_intraday_checkpoints(
            row,
            intraday_bars,
            entry_at,
            entry_price,
            stop_distance,
            side,
            session_close=session.rth_close_at if session else None,
        )
    return row


def _fill_intraday_checkpoints(
    row, intraday_bars, entry_at, entry_price, stop_distance, side, *, session_close=None
):
    """Minute checkpoints after entry, bounded to the entry session.

    ``r_at_eod`` means the entry session's close (7.1: ``r_at_eod`` ≡
    ``entry_r``) - never the last bar of whatever range the caller happened to
    pass. Under a signal-close MOC entry there are no RTH bars after entry, so
    these stay null for swing recipes; that is the honest value.
    """
    ordered = sorted(
        [
            bar
            for bar in intraday_bars
            if bar.get("interval_end") is not None
            and bar["interval_end"] > entry_at
            and (session_close is None or bar["interval_end"] <= session_close)
        ],
        key=lambda bar: bar["interval_end"],
    )
    for column, minutes in INTRADAY_CHECKPOINTS:
        cutoff = entry_at + timedelta(minutes=minutes)
        eligible = [bar for bar in ordered if bar["interval_end"] <= cutoff]
        if eligible:
            row[column] = _r(eligible[-1].get("close"), entry_price, stop_distance, side)
    if ordered:
        row["r_at_eod"] = _r(ordered[-1].get("close"), entry_price, stop_distance, side)


def _tracker_geometry(occurrence: dict) -> list[dict]:
    """The bounded stop list recorded by the tracker adapter."""
    raw = occurrence.get("tags")
    if not isinstance(raw, str) or not raw:
        return []
    try:
        payload = __import__("json").loads(raw)
    except (TypeError, ValueError):
        return []
    candidates = payload.get("stop_candidates") if isinstance(payload, dict) else None
    return [dict(item) for item in (candidates or []) if isinstance(item, dict)]


def _entry_bar_after_d1_close(occurrence: dict, m5_bars) -> tuple[dict | None, object | None]:
    """First completed RTH M5 bar after the D1 setup became known."""
    trigger = occurrence.get("trigger_at")
    if not isinstance(trigger, datetime):
        return None, None
    trigger = trigger if trigger.tzinfo else trigger.replace(tzinfo=timezone.utc)
    ordered = sorted(
        [
            row
            for row in (m5_bars or [])
            if isinstance(row.get("interval_start"), datetime)
            and isinstance(row.get("interval_end"), datetime)
            and row.get("is_complete", True)
            and row["interval_end"] > trigger
        ],
        key=lambda row: row["interval_start"],
    )
    for row in ordered:
        session = xcal.session_for(row["interval_start"])
        if session and session.rth_open_at <= row["interval_start"] < session.rth_close_at:
            return row, session
    return None, None


def _atr14_at_entry(ordered, entry_index: int) -> float | None:
    """Wilder ATR(14) through the completed entry bar, with no later bar."""
    if entry_index < 14:
        return None
    true_ranges: list[float] = []
    for index in range(1, entry_index + 1):
        row = ordered[index]
        previous = ordered[index - 1]
        high = _number(row.get("high"))
        low = _number(row.get("low"))
        prev_close = _number(previous.get("close"))
        if None in (high, low, prev_close):
            continue
        true_ranges.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    if len(true_ranges) < 14:
        return None
    atr = sum(true_ranges[:14]) / 14.0
    for value in true_ranges[14:]:
        atr = ((atr * 13.0) + value) / 14.0
    return atr if atr > 0 else None


def _project_session_close(start: date, count: int) -> datetime | None:
    day = start + timedelta(days=1)
    seen = 0
    for _ in range(60):
        session = xcal.trading_session(day)
        if session is not None:
            seen += 1
            if seen >= count:
                return session.rth_close_at
        day += timedelta(days=1)
    return None


def _selected_tracker_stop(occurrence: dict, recipe: Recipe, entry_price: float, side: str) -> dict | None:
    try:
        source, rank_text = str(recipe.stop_selector).rsplit(":", 1)
        rank = max(1, int(rank_text))
    except (ValueError, TypeError):
        return None
    candidates = []
    for item in _tracker_geometry(occurrence):
        if str(item.get("source_type") or "") != source:
            continue
        level = _number(item.get("level"))
        if level is None:
            continue
        if (side == "LONG" and level >= entry_price) or (side == "SHORT" and level <= entry_price):
            continue
        candidates.append({**item, "level": level})
    candidates.sort(key=lambda item: abs(entry_price - float(item["level"])))
    return candidates[rank - 1] if len(candidates) >= rank else None


def simulate_m5_close_opportunity(
    occurrence: dict,
    m5_bars,
    recipe: Recipe,
    *,
    as_of: datetime,
    computed_at: datetime | None = None,
    run_id: str = "",
) -> dict | None:
    """Approved research entry: next session's first completed M5 close.

    M1 and NBBO are not inputs.  Structural candidates preserve the tracker's
    own close-failure count.  ATR controls use a hard stop.  The primary read
    is STOP_FIRST whenever one M5 range can contain both exits.
    """
    stamp = computed_at or utc_now()
    cutoff = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    side = str(occurrence.get("side") or "").upper()
    if side not in {"LONG", "SHORT"}:
        return None
    ordered = []
    for row in m5_bars or []:
        start = row.get("interval_start")
        end = row.get("interval_end")
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            continue
        end_aware = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        session = xcal.session_for(start)
        if (
            row.get("is_complete", True)
            and end_aware <= cutoff
            and session is not None
            and session.rth_open_at <= start < session.rth_close_at
        ):
            ordered.append(row)
    ordered.sort(key=lambda row: row["interval_start"])
    entry_bar, entry_session = _entry_bar_after_d1_close(occurrence, ordered)
    if entry_bar is None or entry_session is None:
        return None
    entry_index = ordered.index(entry_bar)
    entry_price = _number(entry_bar.get("close"))
    if entry_price is None:
        return None
    hard_stop = recipe.stop_atr_multiple is not None
    close_failures = max(1, int(recipe.close_failures or 1))
    if hard_stop:
        atr = _atr14_at_entry(ordered, entry_index)
        if atr is None:
            return None
        distance = atr * float(recipe.stop_atr_multiple)
        stop_price = entry_price - distance if side == "LONG" else entry_price + distance
    else:
        selected = _selected_tracker_stop(occurrence, recipe, entry_price, side)
        if selected is None:
            return None
        stop_price = float(selected["level"])
        distance = abs(entry_price - stop_price)
        close_failures = max(1, int(_number(selected.get("close_failure_limit")) or 1))
    if distance <= 0 or recipe.target_r is None:
        return None
    direction = 1.0 if side == "LONG" else -1.0
    target_price = entry_price + direction * float(recipe.target_r) * distance
    entry_at = entry_bar["interval_end"]

    future: list[tuple[dict, object]] = []
    session_days: list[date] = []
    post_entry_days: list[date] = []
    for bar in ordered[entry_index + 1 :]:
        session = xcal.session_for(bar["interval_start"])
        if session is None or not (session.rth_open_at <= bar["interval_start"] < session.rth_close_at):
            continue
        if session.session_date not in session_days:
            if (
                session.session_date != entry_session.session_date
                and len(post_entry_days) >= int(recipe.time_stop_sessions or SWING_TIME_STOP_SESSIONS)
            ):
                break
            session_days.append(session.session_date)
            if session.session_date != entry_session.session_date:
                post_entry_days.append(session.session_date)
        future.append((bar, session))

    result_state = STATE_OPEN
    first_hit = None
    first_hit_at = None
    path_resolution = PATH_EXACT
    lower = upper = gross = None
    mfe = mae = 0.0
    time_to_mfe = None
    consecutive = 0
    resolved_at = None
    for bar, _session in future:
        high_r = _r(bar.get("high"), entry_price, distance, side)
        low_r = _r(bar.get("low"), entry_price, distance, side)
        if None not in (high_r, low_r):
            favourable, adverse = max(high_r, low_r), min(high_r, low_r)
            if favourable > mfe:
                mfe = favourable
                time_to_mfe = int((bar["interval_end"] - entry_at).total_seconds() // 60)
            mae = min(mae, adverse)
        target_hit = _reached(bar, target_price, side, favourable=True)
        if hard_stop:
            stop_hit = _reached(bar, stop_price, side, favourable=False)
            stop_exit = stop_price
            open_price = _number(bar.get("open"))
            if open_price is not None and _beyond_stop(open_price, stop_price, side):
                stop_exit = open_price
        else:
            beyond = _beyond_stop(bar.get("close"), stop_price, side)
            consecutive = consecutive + 1 if beyond else 0
            stop_hit = consecutive >= close_failures
            stop_exit = _number(bar.get("close")) or stop_price
        if target_hit and stop_hit:
            result_state = STATE_AMBIGUOUS_BAR
            first_hit = "STOP"
            first_hit_at = resolved_at = bar["interval_end"]
            path_resolution = PATH_AMBIGUOUS
            lower = _r(stop_exit, entry_price, distance, side)
            upper = float(recipe.target_r)
            gross = lower
            break
        if stop_hit:
            result_state = STATE_STOPPED
            first_hit = "STOP"
            first_hit_at = resolved_at = bar["interval_end"]
            gross = _r(stop_exit, entry_price, distance, side)
            break
        if target_hit:
            result_state = STATE_TARGETED
            first_hit = "TARGET"
            first_hit_at = resolved_at = bar["interval_end"]
            gross = float(recipe.target_r)
            break

    time_stop = int(recipe.time_stop_sessions or SWING_TIME_STOP_SESSIONS)
    maturity = _project_session_close(entry_session.session_date, time_stop)
    if result_state == STATE_OPEN and len(post_entry_days) >= time_stop and future:
        result_state = STATE_EXPIRED
        first_hit = "NEITHER"
        resolved_at = future[-1][0]["interval_end"]
        gross = _r(future[-1][0].get("close"), entry_price, distance, side)
    elif result_state == STATE_OPEN and maturity is not None and is_matured({"maturity_at": maturity}, as_of):
        result_state = STATE_TRUNCATED

    row = {
        "occurrence_id": occurrence.get("occurrence_id"),
        "recipe_id": recipe.recipe_id,
        "outcome_definition_id": OUTCOME_DEFINITION_ID,
        "analysis_unit": recipe.analysis_unit,
        "entry_at": entry_at,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "stop_distance": distance,
        "mfe_r": mfe,
        "mae_r": mae,
        "time_to_mfe_min": time_to_mfe,
        "first_hit": first_hit,
        "first_hit_at": first_hit_at,
        "path_resolution": path_resolution,
        "r_lower_bound": lower,
        "r_upper_bound": upper,
        "gross_r": gross,
        "net_r": None if gross is None else net_r(
            gross,
            distance,
            entry_price,
            observed_half_spread=None,
            entry_slippage_half_spreads=ENTRY_SLIPPAGE_HALF_SPREADS,
        ),
        "cost_model_id": OUTCOME_DEFINITION_ID,
        "result_state": result_state,
        "maturity_at": resolved_at or maturity,
        "censor_reason": None,
        "computed_at": stamp,
        "input_capture_mode_worst": _worst_capture_mode(bar.get("capture_mode") for bar in ordered),
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
    }
    for column, minutes in INTRADAY_CHECKPOINTS:
        eligible = [bar for bar, _session in future if bar["interval_end"] <= entry_at + timedelta(minutes=minutes)]
        row[column] = _r(eligible[-1].get("close"), entry_price, distance, side) if eligible else None
    entry_day_bars = [bar for bar, session in future if session.session_date == entry_session.session_date]
    row["r_at_eod"] = (
        _r(entry_day_bars[-1].get("close"), entry_price, distance, side) if entry_day_bars else None
    )
    by_day: dict[date, dict] = {}
    for bar, session in future:
        by_day[session.session_date] = bar
    closes = [by_day[day] for day in sorted(by_day) if day != entry_session.session_date]
    for column, sessions in SWING_CHECKPOINTS:
        row[column] = (
            _r(closes[sessions - 1].get("close"), entry_price, distance, side)
            if len(closes) >= sessions else None
        )
    return row


def _htf_series(
    m5_bars,
    timeframe: str,
    *,
    as_of: datetime,
    exclude_stubs: bool = True,
) -> list[dict]:
    """Rolling multi-session derived bars for one symbol, oldest first.

    Built the way `market_bias_context._derived` builds its rolling series:
    session by session through `aggregate.derive_session_bars`, so the derived
    bars this study reads are the SAME bars the warehouse would publish, under
    the same aggregation contract.

    Stubs are dropped. RTH is 6.5 hours, so H2 and H4 each end a session with a
    short bucket; a 30-minute bar sitting in an H2 series would make the EMA
    measure a duration that changes with the time of day, and completed-bars-
    only means completed bars of the RIGHT length.
    """
    try:  # package import
        from . import aggregate
    except ImportError:  # pragma: no cover - scripts/ directly on sys.path
        import aggregate  # type: ignore

    by_day: dict[date, list[dict]] = {}
    for row in m5_bars or []:
        start = row.get("interval_start")
        if not isinstance(start, datetime) or not row.get("is_complete", True):
            continue
        session = xcal.session_for(start)
        if session is None:
            continue
        by_day.setdefault(session.session_date, []).append(row)
    series: list[dict] = []
    for day in sorted(by_day):
        session = xcal.trading_session(day)
        if session is None:
            continue
        derived = aggregate.derive_session_bars(
            by_day[day], session, timeframe, as_of=as_of, computed_at=as_of
        )
        for row in derived:
            if exclude_stubs and row.get("is_stub"):
                continue
            series.append(row)
    series.sort(key=lambda row: row["interval_end"])
    return series


def _htf_cross_index(series: list[dict], recipe: Recipe, eligible_from: datetime) -> int | None:
    """Index of the first qualifying LRSI cross at or after ``eligible_from``.

    Point-in-time: the oscillator is computed over the whole series, but a
    cross is only usable when the bar that produced it CLOSED after the setup
    became known. A cross printed before then is history the recipe could not
    have traded.
    """
    try:
        from indicators.efficiency_lrsi import compute_efficiency_lrsi
    except ImportError:  # pragma: no cover - indicators ship with the tree
        return None
    closes = [_number(row.get("close")) for row in series]
    if any(value is None for value in closes) or len(closes) < 2:
        return None
    result = compute_efficiency_lrsi([float(value) for value in closes])
    level = float(recipe.cross_level if recipe.cross_level is not None else 50.0)
    indices = (
        result.cross_up_indices(level)
        if recipe.cross_direction == "up"
        else result.cross_down_indices(level)
    )
    for index in indices:
        end = series[index].get("interval_end")
        if isinstance(end, datetime) and end >= eligible_from:
            return index
    return None


def simulate_htf_lrsi_entry(
    occurrence: dict,
    m5_bars,
    recipe: Recipe,
    *,
    as_of: datetime,
    computed_at: datetime | None = None,
    run_id: str = "",
    series_cache: dict | None = None,
) -> dict | None:
    """Phase 0.12 B3: enter on a higher-timeframe LRSI cross. SHADOW ONLY.

    Entry is the CLOSE of the derived bar that printed the cross - a completed
    bar, never a forming one. The stop is that bar's own extreme pushed out a
    quarter of an ATR(14) measured on the same timeframe, and it is a hard
    stop, so the primary read on a bar that could contain both exits is
    STOP_FIRST, exactly as everywhere else in this module.

    Returns ``None`` - no row at all - when the timeframe's series is too short
    to answer, when no qualifying cross has printed yet, or when the ATR is
    unmeasurable. An unanswerable question produces no evidence rather than a
    zero.
    """
    stamp = computed_at or utc_now()
    cutoff = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
    side = "LONG" if recipe.cross_direction == "up" else "SHORT"
    trigger = occurrence.get("trigger_at")
    if not isinstance(trigger, datetime):
        return None
    eligible_from = trigger if trigger.tzinfo else trigger.replace(tzinfo=timezone.utc)

    # The grid runs four entries per timeframe, so without a memo the same
    # rolling series would be rebuilt four times for one occurrence. The cache
    # is per-occurrence and per-cutoff, handed in by the caller and dropped
    # with the occurrence - never a module-level cache that could serve one
    # occurrence's bars to another.
    cache_key = (str(occurrence.get("symbol") or ""), recipe.htf_timeframe, cutoff)
    if series_cache is not None and cache_key in series_cache:
        series = series_cache[cache_key]
    else:
        series = _htf_series(m5_bars, recipe.htf_timeframe, as_of=cutoff)
        series = [row for row in series if row.get("interval_end") <= cutoff]
        if series_cache is not None:
            series_cache[cache_key] = series
    entry_index = _htf_cross_index(series, recipe, eligible_from)
    if entry_index is None:
        return None
    entry_bar = series[entry_index]
    entry_price = _number(entry_bar.get("close"))
    atr = _atr14_at_entry(series, entry_index)
    if entry_price is None or atr is None:
        return None
    extreme = _number(entry_bar.get("low") if side == "LONG" else entry_bar.get("high"))
    if extreme is None:
        return None
    padding = atr * float(recipe.stop_atr_multiple or HTF_LRSI_STOP_ATR_MULTIPLE)
    stop_price = extreme - padding if side == "LONG" else extreme + padding
    distance = abs(entry_price - stop_price)
    if distance <= 0 or recipe.target_r is None:
        return None
    direction = 1.0 if side == "LONG" else -1.0
    target_price = entry_price + direction * float(recipe.target_r) * distance
    entry_at = entry_bar["interval_end"]
    entry_session = xcal.session_for(entry_bar["interval_start"])
    if entry_session is None:
        return None

    time_stop = int(recipe.time_stop_sessions or SWING_TIME_STOP_SESSIONS)
    future: list[tuple[dict, object]] = []
    post_entry_days: list[date] = []
    for bar in series[entry_index + 1 :]:
        session = xcal.session_for(bar["interval_start"])
        if session is None:
            continue
        if (
            session.session_date != entry_session.session_date
            and session.session_date not in post_entry_days
        ):
            if len(post_entry_days) >= time_stop:
                break
            post_entry_days.append(session.session_date)
        future.append((bar, session))

    result_state = STATE_OPEN
    first_hit = None
    first_hit_at = None
    path_resolution = PATH_EXACT
    lower = upper = gross = None
    mfe = mae = 0.0
    time_to_mfe = None
    resolved_at = None
    for bar, _session in future:
        high_r = _r(bar.get("high"), entry_price, distance, side)
        low_r = _r(bar.get("low"), entry_price, distance, side)
        if None not in (high_r, low_r):
            favourable, adverse = max(high_r, low_r), min(high_r, low_r)
            if favourable > mfe:
                mfe = favourable
                time_to_mfe = int((bar["interval_end"] - entry_at).total_seconds() // 60)
            mae = min(mae, adverse)
        target_hit = _reached(bar, target_price, side, favourable=True)
        stop_hit = _reached(bar, stop_price, side, favourable=False)
        stop_exit = stop_price
        open_price = _number(bar.get("open"))
        if open_price is not None and _beyond_stop(open_price, stop_price, side):
            stop_exit = open_price
        if target_hit and stop_hit:
            result_state = STATE_AMBIGUOUS_BAR
            first_hit = "STOP"
            first_hit_at = resolved_at = bar["interval_end"]
            path_resolution = PATH_AMBIGUOUS
            lower = _r(stop_exit, entry_price, distance, side)
            upper = float(recipe.target_r)
            gross = lower
            break
        if stop_hit:
            result_state = STATE_STOPPED
            first_hit = "STOP"
            first_hit_at = resolved_at = bar["interval_end"]
            gross = _r(stop_exit, entry_price, distance, side)
            break
        if target_hit:
            result_state = STATE_TARGETED
            first_hit = "TARGET"
            first_hit_at = resolved_at = bar["interval_end"]
            gross = float(recipe.target_r)
            break

    maturity = _project_session_close(entry_session.session_date, time_stop)
    if result_state == STATE_OPEN and len(post_entry_days) >= time_stop and future:
        result_state = STATE_EXPIRED
        first_hit = "NEITHER"
        resolved_at = future[-1][0]["interval_end"]
        gross = _r(future[-1][0].get("close"), entry_price, distance, side)
    elif result_state == STATE_OPEN and maturity is not None and is_matured(
        {"maturity_at": maturity}, as_of
    ):
        result_state = STATE_TRUNCATED

    row = {
        "occurrence_id": occurrence.get("occurrence_id"),
        "recipe_id": recipe.recipe_id,
        "outcome_definition_id": OUTCOME_DEFINITION_ID,
        "analysis_unit": recipe.analysis_unit,
        "entry_at": entry_at,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "stop_distance": distance,
        "mfe_r": mfe,
        "mae_r": mae,
        "time_to_mfe_min": time_to_mfe,
        "first_hit": first_hit,
        "first_hit_at": first_hit_at,
        "path_resolution": path_resolution,
        "r_lower_bound": lower,
        "r_upper_bound": upper,
        "gross_r": gross,
        "net_r": None if gross is None else net_r(
            gross,
            distance,
            entry_price,
            observed_half_spread=None,
            entry_slippage_half_spreads=ENTRY_SLIPPAGE_HALF_SPREADS,
        ),
        "cost_model_id": OUTCOME_DEFINITION_ID,
        "result_state": result_state,
        "maturity_at": resolved_at or maturity,
        "censor_reason": None,
        "computed_at": stamp,
        "input_capture_mode_worst": _worst_capture_mode(
            bar.get("input_capture_mode_worst") or bar.get("capture_mode") for bar in series
        ),
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
    }
    for column, minutes in INTRADAY_CHECKPOINTS:
        eligible = [
            bar
            for bar, _session in future
            if bar["interval_end"] <= entry_at + timedelta(minutes=minutes)
        ]
        row[column] = (
            _r(eligible[-1].get("close"), entry_price, distance, side) if eligible else None
        )
    entry_day_bars = [
        bar for bar, session in future if session.session_date == entry_session.session_date
    ]
    row["r_at_eod"] = (
        _r(entry_day_bars[-1].get("close"), entry_price, distance, side)
        if entry_day_bars
        else None
    )
    by_day: dict[date, dict] = {}
    for bar, session in future:
        by_day[session.session_date] = bar
    closes = [by_day[day] for day in sorted(by_day) if day != entry_session.session_date]
    for column, sessions in SWING_CHECKPOINTS:
        row[column] = (
            _r(closes[sessions - 1].get("close"), entry_price, distance, side)
            if len(closes) >= sessions
            else None
        )
    return row


def simulate_intraday_bounce(
    occurrence: dict,
    bounce_event: dict,
    m5_bars,
    recipe: Recipe,
    *,
    as_of: datetime,
    session,
    computed_at: datetime | None = None,
    observed_half_spread: float | None = None,
    run_id: str = "",
) -> dict | None:
    """``intraday_bounce_v1`` - only ever from a linked bounce event.

    The bounce event supplies the bounce bar and the production stop; this
    module never decides that a bounce occurred (sec 19.3). The walk is
    bounded to the entry session's RTH bars - the recipe ends at EOD, so a
    bar from any later session is not part of this outcome. A trade whose
    session has not closed yet is OPEN; a session that closed but whose bars
    ran out early is TRUNCATED - neither carries a realized R.
    """
    stamp = computed_at or utc_now()
    side = str(occurrence.get("side") or "LONG").upper()
    bounce_at = bounce_event.get("bounce_at") or bounce_event.get("interval_start")
    stop_price = bounce_event.get("stop_price")
    if bounce_at is None or stop_price is None:
        return None
    ordered = sorted(
        [
            bar
            for bar in m5_bars
            if bar.get("interval_start") is not None
            and session.rth_open_at <= bar["interval_start"] < session.rth_close_at
        ],
        key=lambda bar: bar["interval_start"],
    )
    entry_bar = next((bar for bar in ordered if bar["interval_start"] == bounce_at), None)
    if entry_bar is None:
        return None
    entry_price = float(entry_bar.get("close"))
    stop_level = float(stop_price)
    stop_distance = abs(entry_price - stop_level)
    if not stop_distance:
        return None
    entry_at = entry_bar["interval_end"]
    forward = [bar for bar in ordered if bar["interval_start"] >= entry_at]

    mfe = mae = 0.0
    time_to_mfe = None
    first_hit = None
    first_hit_at = None
    resolved_at = None
    result_state = STATE_OPEN
    exit_r = None
    for bar in forward:
        high_r = _r(bar.get("high"), entry_price, stop_distance, side)
        low_r = _r(bar.get("low"), entry_price, stop_distance, side)
        favourable = max(high_r, low_r) if None not in (high_r, low_r) else None
        adverse = min(high_r, low_r) if None not in (high_r, low_r) else None
        if favourable is not None and favourable > mfe:
            mfe = favourable
            time_to_mfe = int((bar["interval_end"] - entry_at).total_seconds() // 60)
        if adverse is not None and adverse < mae:
            mae = adverse
        if _reached(bar, stop_level, side, favourable=False):
            result_state = STATE_STOPPED
            first_hit = "STOP"
            first_hit_at = resolved_at = bar["interval_end"]
            open_price = _number(bar.get("open"))
            gapped_through = open_price is not None and _beyond_stop(open_price, stop_level, side)
            # A gap through the stop fills at the open, not at the level the
            # market never traded (sec 14.3 gap-through-stop behaviour).
            exit_r = _r(open_price, entry_price, stop_distance, side) if gapped_through else -1.0
            break
    if result_state == STATE_OPEN:
        last_end = forward[-1]["interval_end"] if forward else entry_at
        if last_end >= session.rth_close_at:
            result_state = STATE_EXPIRED  # EOD maturity for intraday
            first_hit = "NEITHER"
            resolved_at = session.rth_close_at
            exit_r = _r(forward[-1].get("close"), entry_price, stop_distance, side)
        elif is_matured({"maturity_at": session.rth_close_at}, as_of):
            # The session closed but the archived bars stop early: truncated
            # evidence, never a finished trade at whatever bar came last.
            result_state = STATE_TRUNCATED
        # else: the session is still trading - the trade is simply OPEN.

    row = {
        "occurrence_id": occurrence.get("occurrence_id"),
        "recipe_id": recipe.recipe_id,
        "outcome_definition_id": OUTCOME_DEFINITION_ID,
        "analysis_unit": recipe.analysis_unit,
        "entry_at": entry_at,
        "entry_price": entry_price,
        "stop_price": stop_level,
        "stop_distance": stop_distance,
        "mfe_r": mfe,
        "mae_r": mae,
        "time_to_mfe_min": time_to_mfe,
        "first_hit": first_hit,
        "first_hit_at": first_hit_at,
        "path_resolution": PATH_EXACT,
        "r_lower_bound": None,
        "r_upper_bound": None,
        "gross_r": exit_r,
        "net_r": None
        if exit_r is None
        else net_r(
            exit_r,
            stop_distance,
            entry_price,
            observed_half_spread=observed_half_spread,
            entry_slippage_half_spreads=ENTRY_SLIPPAGE_HALF_SPREADS,
        ),
        "cost_model_id": OUTCOME_DEFINITION_ID,
        "result_state": result_state,
        # min(EOD, stop): a stopped trade matured when it stopped.
        "maturity_at": resolved_at if resolved_at is not None else session.rth_close_at,
        "censor_reason": None,
        "computed_at": stamp,
        "input_capture_mode_worst": _worst_capture_mode(bar.get("capture_mode") for bar in ordered),
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
    }
    for column, _sessions in SWING_CHECKPOINTS:
        row[column] = None
    for column, _minutes in INTRADAY_CHECKPOINTS:
        row[column] = None
    row["r_at_eod"] = None
    _fill_intraday_checkpoints(
        row, forward, entry_at, entry_price, stop_distance, side, session_close=session.rth_close_at
    )
    return row


def _no_trigger_row(occurrence, recipe, stamp, run_id) -> dict:
    """A setup that never triggered is evidence, not an absence."""
    row = {
        "occurrence_id": occurrence.get("occurrence_id"),
        "recipe_id": recipe.recipe_id,
        "outcome_definition_id": OUTCOME_DEFINITION_ID,
        "analysis_unit": recipe.analysis_unit,
        "entry_at": None,
        "entry_price": None,
        "stop_price": _number(occurrence.get("stop_price_ref")),
        "stop_distance": None,
        "mfe_r": None,
        "mae_r": None,
        "time_to_mfe_min": None,
        "first_hit": None,
        "first_hit_at": None,
        "path_resolution": PATH_EXACT,
        "r_lower_bound": None,
        "r_upper_bound": None,
        "gross_r": None,
        "net_r": None,
        "cost_model_id": OUTCOME_DEFINITION_ID,
        "result_state": STATE_NO_TRIGGER,
        "maturity_at": None,
        "censor_reason": "never triggered",
        "computed_at": stamp,
        "input_capture_mode_worst": "",
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
    }
    for column, _n in SWING_CHECKPOINTS:
        row[column] = None
    for column, _n in INTRADAY_CHECKPOINTS:
        row[column] = None
    row["r_at_eod"] = None
    return row


def _swing_maturity(ordered, entry_index, recipe):
    """The projected time-stop maturity - the LATEST a swing outcome resolves.

    A stop/target/expiry that lands earlier replaces this with its own time
    (sec 14.2's min); this function only answers "when must the answer exist".
    """
    if recipe.time_stop_sessions is None:
        return None, None
    target_index = entry_index + recipe.time_stop_sessions
    if target_index < len(ordered):
        day = _as_date(ordered[target_index]["session_date"])
        session = xcal.trading_session(day)
        return (session.rth_close_at if session else None), target_index
    # Beyond the bars we hold: project forward on the exchange calendar so
    # maturity is a fact about the clock, not about how much data arrived.
    day = _as_date(ordered[entry_index]["session_date"])
    remaining = recipe.time_stop_sessions
    while remaining > 0:
        day += timedelta(days=1)
        if xcal.is_trading_day(day):
            remaining -= 1
    session = xcal.trading_session(day)
    return (session.rth_close_at if session else None), None


def _reached(bar, level, side, *, favourable: bool) -> bool:
    high, low = bar.get("high"), bar.get("low")
    if high is None or low is None or level is None:
        return False
    if (side == "LONG") == favourable:
        return float(high) >= float(level)
    return float(low) <= float(level)


def _beyond_stop(close, level, side) -> bool:
    if close is None or level is None:
        return False
    return float(close) < float(level) if side == "LONG" else float(close) > float(level)


def _session_close(bar):
    day = _as_date(bar.get("session_date"))
    session = xcal.trading_session(day) if day else None
    return session.rth_close_at if session else None


def _as_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return value


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _worst_capture_mode(modes) -> str:
    order = {"LIVE": 0, "DELAYED": 1, "BACKFILL": 2, "RECONSTRUCTED": 3, "": 4}
    worst, rank = "", -1
    for mode in modes:
        value = order.get(str(mode or ""), 4)
        if value > rank:
            rank, worst = value, str(mode or "")
    return worst


# ---------------------------------------------------------------------------
# The read rule and the build job: recompute non-terminal, supersede by time
# ---------------------------------------------------------------------------
def outcome_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("occurrence_id") or ""),
        str(row.get("recipe_id") or ""),
        str(row.get("outcome_definition_id") or ""),
    )


def latest_outcomes(store: ResearchStore, occurrence_ids=None, recipe_ids=None) -> dict:
    """The current view: the latest ``computed_at`` per (occurrence, recipe,
    definition), across every year partition.

    The lake is append-only, so a recomputed outcome is a NEW row that
    supersedes by time; any reader that consumes ``outcome_path`` rows takes
    this view, never the raw row set - a superseded interim reading must not
    coexist with its replacement in anyone's arithmetic.
    """
    latest: dict[tuple[str, str, str], dict] = {}
    for row in store.read_rows(
        "outcome_path", occurrence_ids=occurrence_ids, recipe_ids=recipe_ids
    ):
        key = outcome_key(row)
        current = latest.get(key)
        if current is None or _computed_stamp(row) >= _computed_stamp(current):
            latest[key] = row
    return latest


def _computed_stamp(row) -> datetime:
    value = row.get("computed_at")
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def _same_outcome(previous: dict, computed: dict) -> bool:
    """Would publishing ``computed`` add any knowledge over ``previous``?"""
    for key, value in computed.items():
        if key in ("computed_at", "run_id", "schema_version"):
            continue
        if not _same_value(previous.get(key), value):
            return False
    return True


def _same_value(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        return abs(float(left) - float(right)) < 1e-9
    if isinstance(left, datetime) and isinstance(right, datetime):
        l_value = left if left.tzinfo else left.replace(tzinfo=timezone.utc)
        r_value = right if right.tzinfo else right.replace(tzinfo=timezone.utc)
        return l_value == r_value
    return left == right


def build_outcomes(
    store: ResearchStore | None,
    occurrence_rows,
    *,
    d1_by_symbol=None,
    m5_by_symbol=None,
    bands_by_occurrence=None,
    bounce_by_occurrence=None,
    recipes=None,
    as_of: datetime | None = None,
    now: datetime | None = None,
    run_id: str = "",
    job_id: str = "outcome_path",
) -> OutcomeReport:
    """Simulate the declared recipes for each occurrence.

    Idempotency is by knowledge, not by existence: a terminal row
    (:data:`TERMINAL_RESULT_STATES`) is final evidence and is never
    recomputed; a non-terminal row (OPEN/TRUNCATED/NO_TRIGGER) is
    re-simulated against the bars now available and superseded only when the
    result actually changed. Re-running with the same inputs writes nothing.
    """
    report = OutcomeReport()
    if store is None:
        report.status = "DISABLED"
        return report
    stamp = now or utc_now()
    cutoff = as_of or stamp
    selected = list(recipes or (SWING_HOUSE_V1, CONTROL_FIXED_1R2R_V1, CONTROL_TIME_ONLY_V1))
    occurrence_list = list(occurrence_rows or [])
    identities = [str(row.get("occurrence_id") or "") for row in occurrence_list]
    existing = latest_outcomes(store, identities)

    rows = []
    for occurrence in occurrence_list:
        symbol = str(occurrence.get("symbol") or "")
        # Scoped to this occurrence and dropped with it (B3 derived series).
        htf_series_cache: dict = {}
        for recipe in selected:
            key = (str(occurrence.get("occurrence_id")), recipe.recipe_id, OUTCOME_DEFINITION_ID)
            previous = existing.get(key)
            if previous is not None and str(previous.get("result_state")) in TERMINAL_RESULT_STATES:
                report.skip("ALREADY_SIMULATED")
                continue
            if recipe.timeframe == "D1":
                computed = simulate_swing(
                    occurrence,
                    (d1_by_symbol or {}).get(symbol) or [],
                    recipe,
                    bands=(bands_by_occurrence or {}).get(occurrence.get("occurrence_id")),
                    as_of=cutoff,
                    computed_at=stamp,
                    intraday_bars=(m5_by_symbol or {}).get(symbol),
                    run_id=run_id,
                )
            elif recipe.timeframe == "HTF_LRSI":
                # Shadow lane (Phase 0.12 B3). Sourced from the SAME canonical
                # M5 bars as every other recipe and aggregated through the
                # warehouse's own contract - never a second bar source.
                computed = simulate_htf_lrsi_entry(
                    occurrence,
                    (m5_by_symbol or {}).get(symbol) or [],
                    recipe,
                    as_of=cutoff,
                    computed_at=stamp,
                    run_id=run_id,
                    series_cache=htf_series_cache,
                )
            elif recipe.timeframe == "M5_OPPORTUNITY":
                computed = simulate_m5_close_opportunity(
                    occurrence,
                    (m5_by_symbol or {}).get(symbol) or [],
                    recipe,
                    as_of=cutoff,
                    computed_at=stamp,
                    run_id=run_id,
                )
            else:
                bounce = (bounce_by_occurrence or {}).get(occurrence.get("occurrence_id"))
                if bounce is None:
                    # No linked bounce event: no intraday row is produced, and
                    # nothing is re-detected to manufacture one (sec 19.3).
                    report.skip("NO_LINKED_BOUNCE_EVENT")
                    continue
                trigger = occurrence.get("trigger_at")
                session = xcal.session_for(trigger) if trigger else None
                if session is None:
                    report.skip("NO_SESSION")
                    continue
                computed = simulate_intraday_bounce(
                    occurrence,
                    bounce,
                    (m5_by_symbol or {}).get(symbol) or [],
                    recipe,
                    as_of=cutoff,
                    session=session,
                    computed_at=stamp,
                    run_id=run_id,
                )
            if computed is None:
                report.skip("INSUFFICIENT_PATH_DATA")
                continue
            assert computed["result_state"] in RESULT_STATES, computed["result_state"]
            if previous is not None and _same_outcome(previous, computed):
                # A rebuild that learned nothing writes nothing at all.
                report.skip("UNCHANGED")
                continue
            rows.append(computed)
            report.by_recipe[recipe.recipe_id] = report.by_recipe.get(recipe.recipe_id, 0) + 1

    if not rows:
        report.status = "NOTHING_TO_SIMULATE"
        return report
    report.rows = store.publish("outcome_path", rows, job_id=job_id).rows_published
    return report


__all__ = [
    "ANALYSIS_UNIT_ATTEMPT",
    "ANALYSIS_UNIT_OPPORTUNITY",
    "COMMISSION_PER_SHARE",
    "CONTROL_FIXED_1R2R_V1",
    "CONTROL_TIME_ONLY_V1",
    "DIAGNOSTIC_ATR_STOP_V1",
    "ENTRY_SLIPPAGE_HALF_SPREADS",
    "HALF_SPREAD_BPS",
    "HTF_LRSI_ENTRIES",
    "HTF_LRSI_RECIPES",
    "HTF_LRSI_STOP_ATR_MULTIPLE",
    "HTF_LRSI_TARGETS_R",
    "HTF_LRSI_TIMEFRAMES",
    "INTRADAY_BOUNCE_V1",
    "M5_CLOSE_RECIPES",
    "M5_CLOSE_STOP_RANKS",
    "M5_CLOSE_STOP_SOURCES",
    "M5_CLOSE_TARGETS_R",
    "MIN_HALF_SPREAD",
    "OUTCOME_DEFINITION_ID",
    "PATH_AMBIGUOUS",
    "PATH_EXACT",
    "POST_EARNINGS_CLOSE_FAILURES",
    "PRIMARY_RECIPE_BY_SETUP",
    "RECIPES",
    "SWING_HOUSE_V1",
    "SWING_TIME_STOP_SESSIONS",
    "TERMINAL_RESULT_STATES",
    "OutcomeReport",
    "Recipe",
    "build_outcomes",
    "half_spread",
    "is_matured",
    "latest_outcomes",
    "net_r",
    "outcome_key",
    "simulate_intraday_bounce",
    "simulate_m5_close_opportunity",
    "simulate_swing",
]
