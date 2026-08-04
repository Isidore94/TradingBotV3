"""Recipe simulation and the ``house_default_v1`` outcome contract (Phase 6).

An R number means nothing without the policy that produced it. Every row here
names its ``recipe_id`` (entry, stop, management, time stop) and its
``outcome_definition_id`` (fills, costs, ambiguity, maturity), and alternative
recipes on one occurrence are correlated diagnostics of **one** episode - never
extra samples.

``house_default_v1`` (sec 14.2), implemented exactly:

* ``net_r = gross_r - 2 x (commission_per_share + half_spread) / stop_distance_$``
* commission $0.0035/share (IBKR tiered);
* ``half_spread`` = observed NBBO at signal when supplied, else the declared
  fallback ``max($0.01, 2bp x price)``;
* +1 half_spread of slippage on stop and market entries;
* same-bar ambiguity: **STOP_FIRST is primary**, and the TARGET_FIRST reading
  is retained as ``r_upper_bound`` with ``path_resolution = AMBIGUOUS``;
* maturity: EOD for intraday, ``min(+18 sessions, stop/target/expiry)`` for
  swing.

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


def net_r(gross_r: float, stop_distance: float, price: float, *, observed_half_spread=None) -> float:
    """``gross_r - 2 x (commission + half_spread) / stop_distance`` (sec 14.2)."""
    if not stop_distance:
        return gross_r
    cost = 2.0 * (COMMISSION_PER_SHARE + half_spread(price, observed_half_spread)) / abs(float(stop_distance))
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
    session. Path resolution is honest: when a bar's range contains both the
    stop and the target, STOP_FIRST is the primary estimate and the TARGET_FIRST
    reading is retained as ``r_upper_bound``.
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

    if recipe.stop == "fixed_1r":
        # The control's own risk unit, from the detector's declared geometry.
        declared = abs(entry_price - float(stop_price)) if stop_price is not None else None
        stop_distance = declared
    elif recipe.stop == "none":
        stop_distance = abs(entry_price - float(stop_price)) if stop_price is not None else None
    else:
        stop_distance = abs(entry_price - float(stop_price)) if stop_price is not None else None
    if not stop_distance:
        return None

    close_failures = POST_EARNINGS_CLOSE_FAILURES.get(
        str(occurrence.get("canonical_setup_id") or ""), recipe.close_failures
    )
    direction = 1.0 if side == "LONG" else -1.0
    target_price = None
    if recipe.target_r is not None:
        target_price = entry_price + direction * recipe.target_r * stop_distance
    elif recipe.management.startswith("partial_at_band2") and bands:
        target_price = bands.get("UPPER_3") if side == "LONG" else bands.get("LOWER_3")

    session = xcal.trading_session(trigger_date)
    entry_at = session.rth_close_at if session else trigger_at
    maturity_at, expiry_index = _swing_maturity(ordered, entry_index, recipe)

    result_state = STATE_OPEN
    first_hit = None
    first_hit_at = None
    path_resolution = PATH_EXACT
    r_lower = None
    r_upper = None
    exit_r = None
    consecutive_failures = 0
    mfe = mae = 0.0
    time_to_mfe = None

    # The time stop bounds the path we walk: sessions past it are not part of
    # this recipe's outcome at all.
    horizon = forward[: recipe.time_stop_sessions] if recipe.time_stop_sessions else forward
    for offset, bar in enumerate(horizon, start=1):
        high_r = _r(bar.get("high"), entry_price, stop_distance, side)
        low_r = _r(bar.get("low"), entry_price, stop_distance, side)
        favourable = max(high_r, low_r) if None not in (high_r, low_r) else None
        adverse = min(high_r, low_r) if None not in (high_r, low_r) else None
        if favourable is not None and favourable > mfe:
            mfe = favourable
            time_to_mfe = offset * 24 * 60
        if adverse is not None and adverse < mae:
            mae = adverse

        target_hit = target_price is not None and _reached(bar, target_price, side, favourable=True)
        stop_hit = False
        if recipe.stop != "none":
            close_beyond = _beyond_stop(bar.get("close"), float(stop_price), side)
            consecutive_failures = consecutive_failures + 1 if close_beyond else 0
            stop_hit = consecutive_failures >= close_failures

        if target_hit and stop_hit:
            # Both in one session and OHLC cannot order them: preregistered
            # conservative primary, with the optimistic reading kept as a bound.
            path_resolution = PATH_AMBIGUOUS
            result_state = STATE_AMBIGUOUS_BAR
            first_hit = "STOP"
            first_hit_at = _session_close(bar)
            r_lower = _r(stop_price, entry_price, stop_distance, side)
            r_upper = _r(target_price, entry_price, stop_distance, side)
            exit_r = r_lower
            break
        if stop_hit:
            result_state = STATE_STOPPED
            first_hit = "STOP"
            first_hit_at = _session_close(bar)
            exit_r = _r(bar.get("close"), entry_price, stop_distance, side)
            break
        if target_hit:
            result_state = STATE_TARGETED
            first_hit = "TARGET"
            first_hit_at = _session_close(bar)
            exit_r = _r(target_price, entry_price, stop_distance, side)
            break
    else:
        reached_time_stop = (
            recipe.time_stop_sessions is not None and len(horizon) >= recipe.time_stop_sessions
        )
        if reached_time_stop:
            result_state = STATE_EXPIRED
            first_hit = "NEITHER"
            exit_r = _r(horizon[-1].get("close"), entry_price, stop_distance, side)
        elif horizon:
            # Still running, with the path simply not yet complete.
            result_state = STATE_OPEN
            exit_r = _r(horizon[-1].get("close"), entry_price, stop_distance, side)

    if recipe.management.startswith("partial_at_band2") and bands and result_state != STATE_STOPPED:
        exit_r = _house_management_r(forward, entry_price, stop_distance, side, bands, exit_r)

    if result_state == STATE_OPEN and is_matured({"maturity_at": maturity_at}, as_of):
        # The clock says this should have resolved, but the bars ran out:
        # truncated evidence, never quietly reported as a finished trade.
        result_state = STATE_TRUNCATED

    gross = exit_r
    row = {
        "occurrence_id": occurrence.get("occurrence_id"),
        "recipe_id": recipe.recipe_id,
        "outcome_definition_id": OUTCOME_DEFINITION_ID,
        "analysis_unit": recipe.analysis_unit,
        "entry_at": entry_at,
        "entry_price": entry_price,
        "stop_price": _number(stop_price),
        "stop_distance": stop_distance,
        "mfe_r": mfe,
        "mae_r": mae,
        "time_to_mfe_min": time_to_mfe,
        "first_hit": first_hit,
        "first_hit_at": first_hit_at,
        "path_resolution": path_resolution,
        "r_lower_bound": r_lower,
        "r_upper_bound": r_upper,
        "gross_r": gross,
        "net_r": None if gross is None else net_r(gross, stop_distance, entry_price, observed_half_spread=observed_half_spread),
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
        _fill_intraday_checkpoints(row, intraday_bars, entry_at, entry_price, stop_distance, side)
    return row


def _house_management_r(forward, entry_price, stop_distance, side, bands, fallback):
    """50% partial at band 2, trail to band 1, runner to band 3 (sec 19.3)."""
    band_2 = bands.get("UPPER_2") if side == "LONG" else bands.get("LOWER_2")
    band_1 = bands.get("UPPER_1") if side == "LONG" else bands.get("LOWER_1")
    band_3 = bands.get("UPPER_3") if side == "LONG" else bands.get("LOWER_3")
    if band_2 is None or band_1 is None:
        return fallback
    partial_taken = False
    partial_r = None
    for bar in forward:
        if not partial_taken and _reached(bar, band_2, side, favourable=True):
            partial_taken = True
            partial_r = _r(band_2, entry_price, stop_distance, side)
            continue
        if partial_taken:
            if band_3 is not None and _reached(bar, band_3, side, favourable=True):
                runner_r = _r(band_3, entry_price, stop_distance, side)
                return 0.5 * partial_r + 0.5 * runner_r
            if _beyond_stop(bar.get("close"), band_1, side):
                trail_r = _r(bar.get("close"), entry_price, stop_distance, side)
                return 0.5 * partial_r + 0.5 * trail_r
    if partial_taken and fallback is not None:
        return 0.5 * partial_r + 0.5 * fallback
    return fallback


def _fill_intraday_checkpoints(row, intraday_bars, entry_at, entry_price, stop_distance, side):
    ordered = sorted(
        [bar for bar in intraday_bars if bar.get("interval_end") is not None and bar["interval_end"] > entry_at],
        key=lambda bar: bar["interval_end"],
    )
    for column, minutes in INTRADAY_CHECKPOINTS:
        cutoff = entry_at + timedelta(minutes=minutes)
        eligible = [bar for bar in ordered if bar["interval_end"] <= cutoff]
        if eligible:
            row[column] = _r(eligible[-1].get("close"), entry_price, stop_distance, side)
    if ordered:
        row["r_at_eod"] = _r(ordered[-1].get("close"), entry_price, stop_distance, side)


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
    module never decides that a bounce occurred (sec 19.3).
    """
    stamp = computed_at or utc_now()
    side = str(occurrence.get("side") or "LONG").upper()
    bounce_at = bounce_event.get("bounce_at") or bounce_event.get("interval_start")
    stop_price = bounce_event.get("stop_price")
    if bounce_at is None or stop_price is None:
        return None
    ordered = sorted(
        [bar for bar in m5_bars if bar.get("interval_start") is not None], key=lambda bar: bar["interval_start"]
    )
    entry_bar = next((bar for bar in ordered if bar["interval_start"] == bounce_at), None)
    if entry_bar is None:
        return None
    entry_price = float(entry_bar.get("close"))
    stop_distance = abs(entry_price - float(stop_price))
    if not stop_distance:
        return None
    entry_at = entry_bar["interval_end"]
    forward = [bar for bar in ordered if bar["interval_start"] >= entry_at]

    mfe = mae = 0.0
    time_to_mfe = None
    first_hit = None
    first_hit_at = None
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
        if _reached(bar, float(stop_price), side, favourable=False):
            result_state = STATE_STOPPED
            first_hit = "STOP"
            first_hit_at = bar["interval_end"]
            exit_r = -1.0
            break
    if result_state == STATE_OPEN and forward:
        result_state = STATE_EXPIRED  # EOD maturity for intraday
        first_hit = "NEITHER"
        exit_r = _r(forward[-1].get("close"), entry_price, stop_distance, side)

    row = {
        "occurrence_id": occurrence.get("occurrence_id"),
        "recipe_id": recipe.recipe_id,
        "outcome_definition_id": OUTCOME_DEFINITION_ID,
        "analysis_unit": recipe.analysis_unit,
        "entry_at": entry_at,
        "entry_price": entry_price,
        "stop_price": float(stop_price),
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
        else net_r(exit_r, stop_distance, entry_price, observed_half_spread=observed_half_spread),
        "cost_model_id": OUTCOME_DEFINITION_ID,
        "result_state": result_state,
        "maturity_at": session.rth_close_at,
        "censor_reason": None,
        "computed_at": stamp,
        "input_capture_mode_worst": _worst_capture_mode(bar.get("capture_mode") for bar in ordered),
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
    }
    for column, _sessions in SWING_CHECKPOINTS:
        row[column] = None
    _fill_intraday_checkpoints(row, forward, entry_at, entry_price, stop_distance, side)
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
    """Simulate the declared recipes for each occurrence. Idempotent."""
    report = OutcomeReport()
    if store is None:
        report.status = "DISABLED"
        return report
    stamp = now or utc_now()
    cutoff = as_of or stamp
    selected = list(recipes or (SWING_HOUSE_V1, CONTROL_FIXED_1R2R_V1, CONTROL_TIME_ONLY_V1))

    existing = set()
    years = {row.get("event_at").year for row in occurrence_rows or [] if row.get("event_at")}
    for year in sorted(years or {cutoff.year}):
        for row in store.read_table(
            "outcome_path", f"year={year}", columns=["occurrence_id", "recipe_id", "outcome_definition_id"]
        ).to_pylist():
            existing.add((str(row["occurrence_id"]), str(row["recipe_id"]), str(row["outcome_definition_id"])))

    rows = []
    for occurrence in occurrence_rows or []:
        symbol = str(occurrence.get("symbol") or "")
        for recipe in selected:
            key = (str(occurrence.get("occurrence_id")), recipe.recipe_id, OUTCOME_DEFINITION_ID)
            if key in existing:
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
    "HALF_SPREAD_BPS",
    "INTRADAY_BOUNCE_V1",
    "MIN_HALF_SPREAD",
    "OUTCOME_DEFINITION_ID",
    "PATH_AMBIGUOUS",
    "PATH_EXACT",
    "POST_EARNINGS_CLOSE_FAILURES",
    "PRIMARY_RECIPE_BY_SETUP",
    "RECIPES",
    "SWING_HOUSE_V1",
    "SWING_TIME_STOP_SESSIONS",
    "OutcomeReport",
    "Recipe",
    "build_outcomes",
    "half_spread",
    "is_matured",
    "net_r",
    "simulate_intraday_bounce",
    "simulate_swing",
]
