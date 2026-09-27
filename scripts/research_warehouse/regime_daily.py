"""Point-in-time daily market regimes for SPY/QQQ/IWM: gold dataset ``market_regime_daily``.

One row per (symbol, session_date, rule_version). Every label on a row is computed
ONLY from bars completed by that session's close, so a row is usable for trading the
NEXT session (``next_session_date``): a backtest trading on session T joins the row
whose ``next_session_date == T`` (``read_regimes(..., as_of="next_open")`` does this).

Axes, each kept in its own column (missing input = ``unknown``, never a guess):

* ``env_d1`` / ``env_w``: the champion Auto Market Bias env_key, called through
  ``market_regimes.d1_env_key`` / ``weekly_env_key`` exactly as the S17 table does,
  read as of the next session's open (= bars up to this close).
* ``env_h1`` / ``env_h4``: the same champion read over the last 20 completed H1/H4
  bars (as ``market_bias_context.context_at`` does); ``unknown`` when that session
  has no intraday bars (only ~2 years of H1 exist).
* ``trend20``: ``long_lab.spy_trend_labels`` (above_rising_20d / above_falling_20d /
  below_20d). ``trend50_200``: uptrend / downtrend / transition.
* ``vol_rv``: 20-session realized vol percentile vs the trailing 252 sessions.
  ``vol_vix``: ^VIX close bucket.
* ``drawdown``: % off the 252-session high.
* ``structural`` (``auto_structural_v1``): the trader's vocabulary from
  ``structural_regime.VOCABULARY`` by fixed rules, with causal hysteresis;
  ``structural_raw`` is the unsmoothed rule output.
* ``composite``: ``trend50_200|vol_rv`` for painting setups.

Thresholds are named constants. Changing any rule means a new ``RULE_VERSION``; old
rows are never rewritten. Sessions inside the warm-up (fewer than ``WARMUP_SESSIONS``
bars of history) are not written, so earlier history arriving later can still fill them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

DATASET = "market_regime_daily"
RULE_VERSION = "regime_daily_v1"
#: Oldest first; ``read_regimes`` prefers the newest version present.
RULE_HISTORY: tuple[str, ...] = (RULE_VERSION,)
STRUCTURAL_RULE_VERSION = "auto_structural_v1"
UNKNOWN = "unknown"
MARKET_TZ = ZoneInfo("America/New_York")

SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "IWM")
VIX_SYMBOL = "^VIX"
HISTORY_START = date(2018, 1, 1)
#: A session is final for this job once this ET clock time has passed.
CLOSE_SETTLED_AT = time(16, 15)

# ---- trend
SMA_FAST = 50
SMA_SLOW = 200
#: The 200-day SMA is rising when above its value this many sessions earlier.
SMA_SLOW_SLOPE_SESSIONS = 20
#: The 50-day SMA is falling when below its value this many sessions earlier.
SMA_FAST_SLOPE_SESSIONS = 10
TREND_UP, TREND_DOWN, TREND_TRANSITION = "uptrend", "downtrend", "transition"

# ---- volatility
RV_WINDOW = 20
RV_RANK_WINDOW = 252
TRADING_DAYS = 252
#: (upper bound of the percentile, label); at or above the last bound is "extreme".
RV_CUTS: tuple[tuple[float, str], ...] = ((0.25, "low"), (0.75, "normal"), (0.95, "high"))
RV_TOP = "extreme"
#: (upper bound of the ^VIX close, label); at or above the last bound is the top label.
VIX_CUTS: tuple[tuple[float, str], ...] = ((15.0, "vix_lt15"), (20.0, "vix_15_20"), (30.0, "vix_20_30"))
VIX_TOP = "vix_30_plus"

# ---- drawdown
DD_WINDOW = 252
#: (upper bound in percent off the 252-session high, label).
DD_CUTS: tuple[tuple[float, str], ...] = ((3.0, "dd_0_3"), (8.0, "dd_3_8"), (15.0, "dd_8_15"))
DD_TOP = "dd_15_plus"

# ---- auto structural v1
WEEKLY_SWING_WEEKS = 4
WEEKS_NEEDED = 26
COMPRESSION_RECENT_WEEKS = 3
COMPRESSION_BASE_WEEKS = 8
COMPRESSION_RATIO = 0.75
RECENT_HIGH_WEEKS = 8
PAUSE_WEEKS = 2
RET_SHORT_SESSIONS = 5
CAPITULATION_MIN_DD = 0.08
CAPITULATION_MAX_RET5 = -0.05
RECOVERY_LOOKBACK = 63
RECOVERY_MIN_PRIOR_DD = 0.10
RECOVERY_MIN_BOUNCE = 0.07
BULL_MAX_DD = 0.05
#: A new structural label must hold this many sessions before it replaces the old one.
HYSTERESIS_SESSIONS = 3
#: Labels that switch at once (events, not states).
FAST_LABELS = frozenset({"capitulation"})
HYSTERESIS_LOOKBACK = 126

COMPOSITE_KEY = "trend50_200|vol_rv"

#: Bars a session needs behind it before its row is written: the longest input is the
#: recovery rule's 63-session max of the 252-session drawdown.
WARMUP_SESSIONS = max(RV_WINDOW + RV_RANK_WINDOW, DD_WINDOW + RECOVERY_LOOKBACK - 1)
#: Intraday champion window, as in ``market_bias_context.ROLLING_BARS``.
INTRADAY_WINDOW = 20
#: Hold a missing ^VIX / intraday session back this many newest sessions before writing it unknown.
HOLD_SESSIONS = 2

AXES: tuple[str, ...] = (
    "env_d1", "env_w", "env_h4", "env_h1", "trend20", "trend50_200",
    "vol_rv", "vol_vix", "drawdown", "structural", "structural_raw", "composite",
)


# ---------------------------------------------------------------- inputs
def _as_day(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "date") and callable(value.date):  # pandas Timestamp
        try:
            return value.date()
        except (TypeError, ValueError):
            return None
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _num(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _records(source: Any) -> list[dict]:
    if source is None:
        return []
    if hasattr(source, "to_dict"):
        return list(source.to_dict("records"))
    return [dict(row) for row in source]


def daily_bars(source: Any) -> list[dict]:
    """Clean D1 rows, oldest first, one per session: dropped when OHLC is missing or not positive."""
    by_day: dict[date, dict] = {}
    for row in _records(source):
        day = _as_day(row.get("session_date", row.get("date")))
        values = {key: _num(row.get(key)) for key in ("open", "high", "low", "close")}
        if day is None or any(value is None or value <= 0 for value in values.values()):
            continue
        by_day[day] = {"session_date": day, **values, "volume": _num(row.get("volume")) or 0.0}
    return [by_day[day] for day in sorted(by_day)]


def _aware(value: Any) -> datetime | None:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else None


def intraday_bars(source: Any) -> list[dict]:
    """Completed, non-stub intraday rows with a tz-aware ``interval_start``, oldest first, by ET session date."""
    out: dict[datetime, dict] = {}
    for row in _records(source):
        start = _aware(row.get("interval_start"))
        close = _num(row.get("close"))
        if start is None or close is None or row.get("is_complete") in (False,) or row.get("is_stub") in (True,):
            continue
        out[start] = {
            **{key: _num(row.get(key)) for key in ("open", "high", "low", "volume")},
            "close": close,
            "interval_start": start,
            "session_date": start.astimezone(MARKET_TZ).date(),
        }
    return [out[key] for key in sorted(out)]


def next_session_after(day: date) -> date:
    from research_warehouse import exchange_calendar as xcal

    probe = day + timedelta(days=1)
    while not xcal.is_trading_day(probe):
        probe += timedelta(days=1)
    return probe


def last_settled_session(now: datetime) -> date:
    """The newest session whose close has settled at ``now`` (ET 16:15 on a trading day)."""
    from research_warehouse import exchange_calendar as xcal

    local = now.astimezone(MARKET_TZ)
    day = local.date()
    if not (xcal.is_trading_day(day) and local.time() >= CLOSE_SETTLED_AT):
        day -= timedelta(days=1)
        while not xcal.is_trading_day(day):
            day -= timedelta(days=1)
    return day


# ---------------------------------------------------------------- pure labels
def bucket(value: float | None, cuts: Sequence[tuple[float, str]], top: str) -> str:
    if value is None or not math.isfinite(value):
        return UNKNOWN
    for bound, name in cuts:
        if value < bound:
            return name
    return top


def trend50_200(close: float | None, sma50: float | None, sma200: float | None, sma200_prior: float | None) -> str:
    if None in (close, sma50, sma200, sma200_prior):
        return UNKNOWN
    if close > sma50 and close > sma200 and sma200 > sma200_prior:
        return TREND_UP
    if close < sma200 and sma200 < sma200_prior:
        return TREND_DOWN
    return TREND_TRANSITION


def composite(trend: str, vol: str) -> str:
    return UNKNOWN if UNKNOWN in (trend, vol) else f"{trend}|{vol}"


@dataclass
class StructuralFacts:
    close: float | None = None
    sma50: float | None = None
    sma50_prior: float | None = None
    sma200: float | None = None
    sma200_prior: float | None = None
    drawdown: float | None = None  # fraction off the 252-session high
    max_drawdown_recent: float | None = None
    bounce: float | None = None  # close vs the lowest close of the recovery lookback
    ret5: float | None = None
    weekly: list[dict] = field(default_factory=list)  # finished weeks, oldest first


def weekly_swing(weeks: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Last ``WEEKLY_SWING_WEEKS`` finished weeks against the ones before: HH/HL/LH/LL."""
    n = WEEKLY_SWING_WEEKS
    if len(weeks) < 2 * n:
        return None
    recent, prior = weeks[-n:], weeks[-2 * n:-n]
    rh, ph = max(w["high"] for w in recent), max(w["high"] for w in prior)
    rl, pl = min(w["low"] for w in recent), min(w["low"] for w in prior)
    return {"hh": rh > ph, "lh": rh < ph, "hl": rl > pl, "ll": rl < pl}


def swing_text(swing: Mapping[str, Any] | None) -> str:
    if swing is None:
        return UNKNOWN
    highs = "hh" if swing["hh"] else "lh" if swing["lh"] else "eh"
    lows = "hl" if swing["hl"] else "ll" if swing["ll"] else "el"
    return f"{highs}_{lows}"


def range_ratio(weeks: Sequence[Mapping[str, Any]]) -> float | None:
    """Mean range of the last 3 finished weeks over the mean of the 8 weeks before them."""
    need = COMPRESSION_RECENT_WEEKS + COMPRESSION_BASE_WEEKS
    if len(weeks) < need:
        return None
    tail = weeks[-need:]
    recent = [w["high"] - w["low"] for w in tail[-COMPRESSION_RECENT_WEEKS:]]
    base = [w["high"] - w["low"] for w in tail[:COMPRESSION_BASE_WEEKS]]
    base_mean = sum(base) / len(base)
    return (sum(recent) / len(recent)) / base_mean if base_mean > 0 else None


def structural_raw(facts: StructuralFacts) -> str:
    """auto_structural_v1 for one session, first matching rule wins:

    capitulation: >= 8% off the high and the 5-session return <= -5%.
    recovery: a >= 10% drawdown inside the last 63 sessions, >= 7% off that stretch's
      lowest close, and the 50/200 trend not yet an uptrend.
    bear_channel_lower_highs: weekly lower highs and lower lows, and below a falling
      50-day or below a falling 200-day.
    bull_run: uptrend (above 50 and 200, 200 rising), weekly higher highs and higher
      lows, < 5% off the high, weekly ranges not contracting.
    weekly_hh_then_compression: above a rising 200-day, a 26-week high inside the last
      8 weeks, none in the last 2, and weekly ranges contracted (ratio <= 0.75).
    range: every other session with known inputs.
    """
    f = facts
    needed = (f.close, f.sma50, f.sma50_prior, f.sma200, f.sma200_prior, f.drawdown,
              f.max_drawdown_recent, f.bounce, f.ret5)
    if any(value is None for value in needed) or len(f.weekly) < WEEKS_NEEDED:
        return UNKNOWN
    swing = weekly_swing(f.weekly)
    ratio = range_ratio(f.weekly)
    if swing is None or ratio is None:
        return UNKNOWN
    trend = trend50_200(f.close, f.sma50, f.sma200, f.sma200_prior)
    contracting = ratio <= COMPRESSION_RATIO
    if f.drawdown >= CAPITULATION_MIN_DD and f.ret5 <= CAPITULATION_MAX_RET5:
        return "capitulation"
    if f.max_drawdown_recent >= RECOVERY_MIN_PRIOR_DD and f.bounce >= RECOVERY_MIN_BOUNCE and trend != TREND_UP:
        return "recovery"
    below_falling_50 = f.close < f.sma50 and f.sma50 < f.sma50_prior
    below_falling_200 = f.close < f.sma200 and f.sma200 < f.sma200_prior
    if swing["lh"] and swing["ll"] and (below_falling_50 or below_falling_200):
        return "bear_channel_lower_highs"
    if trend == TREND_UP and swing["hh"] and swing["hl"] and f.drawdown < BULL_MAX_DD and not contracting:
        return "bull_run"
    highs = [w["high"] for w in f.weekly]
    recent_high = max(highs[-RECENT_HIGH_WEEKS:]) >= max(highs[-WEEKS_NEEDED:])
    paused = max(highs[-PAUSE_WEEKS:]) < max(highs[-RECENT_HIGH_WEEKS:])
    if f.close > f.sma200 and f.sma200 > f.sma200_prior and recent_high and paused and contracting:
        return "weekly_hh_then_compression"
    return "range"


def smooth_labels(raw: Sequence[str]) -> list[str]:
    """Causal hysteresis: the label of the newest run that reached its hold length by that session.

    A run of a non-fast label must last ``HYSTERESIS_SESSIONS`` sessions to take over;
    ``FAST_LABELS`` take over at once. Looks back at most ``HYSTERESIS_LOOKBACK``
    sessions; no confirmed run there (or only unknowns) is ``unknown``.
    """
    run = [0] * len(raw)
    for i, label in enumerate(raw):
        run[i] = run[i - 1] + 1 if i and raw[i - 1] == label else 1
    out: list[str] = []
    for i in range(len(raw)):
        chosen = UNKNOWN
        for j in range(i, max(-1, i - HYSTERESIS_LOOKBACK - 1), -1):
            label = raw[j]
            need = 1 if label in FAST_LABELS else HYSTERESIS_SESSIONS
            if label != UNKNOWN and run[j] >= need:
                chosen = label
                break
            if label == UNKNOWN and run[j] >= HYSTERESIS_SESSIONS:
                break  # a confirmed unknown stretch: nothing older carries over
        out.append(chosen)
    return out


# ---------------------------------------------------------------- series maths
def _trailing(values: np.ndarray, window: int, how: Callable) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) >= window:
        view = np.lib.stride_tricks.sliding_window_view(values, window)
        out[window - 1:] = how(view, axis=1)
    return out


def _at(values: np.ndarray, i: int) -> float | None:
    if i < 0 or i >= len(values):
        return None
    value = float(values[i])
    return value if math.isfinite(value) else None


def realized_vol(close: np.ndarray) -> np.ndarray:
    """Annualized stdev of the last ``RV_WINDOW`` log returns (needs RV_WINDOW + 1 closes)."""
    rets = np.full(len(close), np.nan)
    if len(close) > 1:
        rets[1:] = np.log(close[1:] / close[:-1])
    out = np.full(len(close), np.nan)
    if len(close) > RV_WINDOW:
        view = np.lib.stride_tricks.sliding_window_view(rets[1:], RV_WINDOW)
        out[RV_WINDOW:] = np.std(view, axis=1, ddof=1) * math.sqrt(TRADING_DAYS)
    return out


def rv_percentile(rv: np.ndarray) -> np.ndarray:
    """Share of the trailing ``RV_RANK_WINDOW`` rv values (today included) at or below today's."""
    out = np.full(len(rv), np.nan)
    if len(rv) >= RV_RANK_WINDOW:
        view = np.lib.stride_tricks.sliding_window_view(rv, RV_RANK_WINDOW)
        today = view[:, -1:]
        full = np.all(np.isfinite(view), axis=1)
        share = np.sum(view <= today, axis=1) / RV_RANK_WINDOW
        out[RV_RANK_WINDOW - 1:] = np.where(full, share, np.nan)
    return out


# ---------------------------------------------------------------- env keys
def _weekly_slice_start(bars: Sequence[Mapping[str, Any]], i: int, weeks: int = 40) -> int:
    """First index of a Monday-aligned stretch ``weeks`` weeks before bar ``i`` (whole weeks only)."""
    day = bars[i]["session_date"]
    floor = day - timedelta(days=day.weekday()) - timedelta(weeks=weeks)
    j = i
    while j > 0 and bars[j - 1]["session_date"] >= floor:
        j -= 1
    return j


def intraday_env_key(bars: Sequence[Mapping[str, Any]], day: date) -> str:
    """Champion env_key over the last 20 intraday bars completed by ``day``'s close; needs bars on ``day``."""
    if not bars or bars[-1]["session_date"] != day or len(bars) <= INTRADAY_WINDOW:
        return UNKNOWN
    from research_warehouse import market_bias_context as bias

    reference = _num(bars[-INTRADAY_WINDOW - 1].get("close"))
    reading = bias._champion_read([dict(row) for row in bars[-INTRADAY_WINDOW:]], reference)
    return str(reading.get("env_key") or UNKNOWN)


def _intraday_upto(bars: Sequence[Mapping[str, Any]], days: Sequence[date]) -> list[int]:
    """For each day, the count of bars whose session is on or before it."""
    out, j = [], 0
    for day in days:
        while j < len(bars) and bars[j]["session_date"] <= day:
            j += 1
        out.append(j)
    return out


# ---------------------------------------------------------------- rows
def build_rows(
    symbol: str,
    d1: Any,
    *,
    vix: Any = None,
    h1: Any = None,
    h4: Any = None,
    computed_at: datetime | None = None,
    run_id: str = "",
    source: str = "",
    since: date | None = None,
    until: date | None = None,
) -> list[dict]:
    """Every post-warm-up session of ``symbol`` in [since, until] as one regime row."""
    import market_regimes as mr
    from market_structure import completed_weekly_bars
    from research_warehouse import long_lab
    from research_warehouse.schemas import SCHEMA_VERSION

    bars = daily_bars(d1)
    if until is not None:
        bars = [bar for bar in bars if bar["session_date"] <= until]
    if len(bars) < WARMUP_SESSIONS:
        return []
    stamp = (computed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    days = [bar["session_date"] for bar in bars]
    series = long_lab.Series.from_rows(symbol, [{**bar, "date": bar["session_date"]} for bar in bars])
    trend20 = long_lab.spy_trend_labels(series)
    close, high = series.close, series.high
    sma20, sma50, sma200 = series.sma20, series.sma50, series.sma200
    rv = realized_vol(close)
    rv_pct = rv_percentile(rv)
    peak = _trailing(high, DD_WINDOW, np.max)
    dd = 1.0 - close / peak
    max_dd_recent = _trailing(dd, RECOVERY_LOOKBACK, np.max)
    low_recent = _trailing(close, RECOVERY_LOOKBACK, np.min)
    vix_close = {bar["session_date"]: bar["close"] for bar in daily_bars(vix)}
    h1_bars, h4_bars = intraday_bars(h1), intraday_bars(h4)
    h1_upto, h4_upto = _intraday_upto(h1_bars, days), _intraday_upto(h4_bars, days)
    weeks_all = completed_weekly_bars(bars, days[-1] + timedelta(days=14))

    nexts = [days[i + 1] if i + 1 < len(days) else next_session_after(days[i]) for i in range(len(days))]
    raw: list[str] = []
    facts_by_i: list[tuple[StructuralFacts, float | None]] = []
    week_index = 0
    for i in range(len(days)):
        while week_index < len(weeks_all) and weeks_all[week_index]["week_start"] + timedelta(days=7) <= nexts[i]:
            week_index += 1
        weeks = weeks_all[:week_index]
        facts = StructuralFacts(
            close=_at(close, i),
            sma50=_at(sma50, i),
            sma50_prior=_at(sma50, i - SMA_FAST_SLOPE_SESSIONS),
            sma200=_at(sma200, i),
            sma200_prior=_at(sma200, i - SMA_SLOW_SLOPE_SESSIONS),
            drawdown=_at(dd, i),
            max_drawdown_recent=_at(max_dd_recent, i),
            bounce=(close[i] / low_recent[i] - 1.0) if _at(low_recent, i) else None,
            ret5=(close[i] / close[i - RET_SHORT_SESSIONS] - 1.0) if i >= RET_SHORT_SESSIONS else None,
            weekly=weeks[-WEEKS_NEEDED - 4:],
        )
        facts_by_i.append((facts, range_ratio(weeks)))
        raw.append(structural_raw(facts))
    smooth = smooth_labels(raw)

    rows: list[dict] = []
    for i, day in enumerate(days):
        if i < WARMUP_SESSIONS - 1 or (since is not None and day < since):
            continue
        facts, ratio = facts_by_i[i]
        env_d1 = mr.d1_env_key(bars[max(0, i - mr.D1_WINDOW - 5): i + 1], mr.D1_WINDOW)
        env_w = mr.weekly_env_key(bars[_weekly_slice_start(bars, i): i + 1], nexts[i])
        vol_rv = bucket(_at(rv_pct, i), RV_CUTS, RV_TOP)
        t50 = trend50_200(facts.close, facts.sma50, facts.sma200, facts.sma200_prior)
        drawdown_pct = None if facts.drawdown is None else facts.drawdown * 100.0
        rows.append(
            {
                "symbol": symbol,
                "session_date": day,
                "rule_version": RULE_VERSION,
                "next_session_date": nexts[i],
                "env_d1": env_d1,
                "env_w": env_w,
                "env_h4": intraday_env_key(h4_bars[: h4_upto[i]], day),
                "env_h1": intraday_env_key(h1_bars[: h1_upto[i]], day),
                "trend20": trend20.get(day, UNKNOWN),
                "trend50_200": t50,
                "vol_rv": vol_rv,
                "vol_vix": bucket(vix_close.get(day), VIX_CUTS, VIX_TOP),
                "drawdown": bucket(drawdown_pct, DD_CUTS, DD_TOP),
                "structural_raw": raw[i],
                "structural": smooth[i],
                "structural_rule_version": STRUCTURAL_RULE_VERSION,
                "composite": composite(t50, vol_rv),
                "composite_key": COMPOSITE_KEY,
                "close": facts.close,
                "sma20": _at(sma20, i),
                "sma50": facts.sma50,
                "sma200": facts.sma200,
                "rv20": _at(rv, i),
                "rv20_pct": _at(rv_pct, i),
                "vix_close": vix_close.get(day),
                "drawdown_pct": drawdown_pct,
                "ret5_pct": None if facts.ret5 is None else facts.ret5 * 100.0,
                "weekly_swing": swing_text(weekly_swing(facts.weekly)),
                "weekly_range_ratio": ratio,
                "d1_bar_count": i + 1,
                "bars_source": source,
                "computed_at": stamp,
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
            }
        )
    return rows


# ---------------------------------------------------------------- lake
def existing_keys(store, symbols: Iterable[str]) -> set[tuple[str, date, str]]:
    rows = store.read_rows(DATASET, columns=["symbol", "session_date", "rule_version"], symbols=list(symbols))
    return {(str(r["symbol"]), _as_day(r["session_date"]), str(r["rule_version"])) for r in rows}


def new_rows(store, rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Rows whose (symbol, session_date, rule_version) is not in the lake yet."""
    have = existing_keys(store, {str(row["symbol"]) for row in rows}) if rows else set()
    return [dict(row) for row in rows if (row["symbol"], row["session_date"], row["rule_version"]) not in have]


def _default_d1_loader(symbols, start, end, *, store=None):
    from research_warehouse import history_reader

    return history_reader.read_d1(list(symbols), start, end, store=store)


def _default_intraday_loader(timeframe, symbols, start, end, *, store=None):
    from research_warehouse import history_reader

    return history_reader.read_intraday(timeframe, list(symbols), start, end, store=store)


def run_build(
    store,
    *,
    symbols: Sequence[str] = SYMBOLS,
    apply: bool = False,
    now: datetime | None = None,
    until: date | None = None,
    d1_loader: Callable | None = None,
    intraday_loader: Callable | None = None,
    run_id: str = "",
) -> dict:
    """Compute every missing regime row and (with ``apply``) publish it. Idempotent.

    The newest ``HOLD_SESSIONS`` sessions wait while their ^VIX close (or, for a
    symbol that has recent intraday bars, their H1 bars) is not in yet: rows are
    never rewritten, so writing them early would fix an ``unknown`` for good.
    """
    if store is None:
        return {"status": "DISABLED", "message": "research_store_dir is not configured."}
    stamp = now or datetime.now(timezone.utc)
    last = min(until, last_settled_session(stamp)) if until else last_settled_session(stamp)
    # Default loaders read bars from the same lake the rows are written to.
    load_d1 = d1_loader or (lambda symbols, start, end: _default_d1_loader(symbols, start, end, store=store))
    load_intraday = intraday_loader or (
        lambda timeframe, symbols, start, end: _default_intraday_loader(timeframe, symbols, start, end, store=store)
    )
    wanted = [str(symbol).upper() for symbol in symbols]
    d1_by_symbol = load_d1(wanted + [VIX_SYMBOL], HISTORY_START, last) or {}
    intraday: dict[str, dict[str, Any]] = {}
    notes: list[str] = []
    for timeframe in ("H1", "H4"):
        try:
            intraday[timeframe] = load_intraday(timeframe, wanted, last - timedelta(days=800), last) or {}
        except Exception as exc:  # noqa: BLE001 - no intraday history is unknown H1/H4, never a failure
            intraday[timeframe] = {}
            notes.append(f"{timeframe} unreadable ({type(exc).__name__}); env_{timeframe.lower()} unknown")
    vix = d1_by_symbol.get(VIX_SYMBOL)
    vix_days = {bar["session_date"] for bar in daily_bars(vix)}
    candidates: list[dict] = []
    held: list[str] = []
    for symbol in wanted:
        rows = build_rows(
            symbol, d1_by_symbol.get(symbol), vix=vix,
            h1=intraday["H1"].get(symbol), h4=intraday["H4"].get(symbol),
            computed_at=stamp, run_id=run_id, source="history_reader", until=last,
        )
        if not rows:
            continue
        newest = [row["session_date"] for row in rows[-HOLD_SESSIONS:]]
        h1_days = {bar["session_date"] for bar in intraday_bars(intraday["H1"].get(symbol))}
        recent_h1 = bool(h1_days) and max(h1_days) >= newest[0] - timedelta(days=14)
        for row in rows:
            day = row["session_date"]
            if day in newest and (day not in vix_days or (recent_h1 and day not in h1_days)):
                held.append(f"{symbol} {day.isoformat()}")
                continue
            candidates.append(row)
    fresh = new_rows(store, candidates)
    report: dict[str, Any] = {
        "status": "OK",
        "applied": bool(apply),
        "dataset": DATASET,
        "rule_version": RULE_VERSION,
        "last_session": last.isoformat(),
        "computed": len(candidates),
        "new_rows": len(fresh),
        "by_symbol": {symbol: sum(1 for row in fresh if row["symbol"] == symbol) for symbol in wanted},
        "held": held,
        "notes": notes,
        "unknown": {
            axis: sum(1 for row in fresh if row.get(axis) == UNKNOWN) for axis in AXES
        },
    }
    if fresh:
        report["first_session"] = min(row["session_date"] for row in fresh).isoformat()
    if apply and fresh:
        result = store.publish(DATASET, fresh, job_id=DATASET)
        report["rows_published"] = result.rows_published
        report["rows_quarantined"] = result.rows_quarantined
        if result.rows_quarantined:
            report["status"] = "PARTIAL"
    return report


# ---------------------------------------------------------------- reader
def _rank(version: Any) -> int:
    try:
        return RULE_HISTORY.index(str(version))
    except ValueError:
        return -1


def read_regimes(
    symbol: str = "SPY",
    rule_version: str | None = None,
    *,
    store=None,
    as_of: str = "close",
):
    """One symbol's regime rows as a DataFrame sorted by ``session_date`` (python dates).

    ``rule_version=None`` picks the newest version in ``RULE_HISTORY`` that has rows.
    ``as_of="close"`` keys each row by the session whose close it describes;
    ``as_of="next_open"`` adds ``trade_session_date`` = ``next_session_date``, the
    session a backtest may use the label for. Read-only; an unset lake is an empty frame.
    """
    import pandas as pd

    from research_warehouse.schemas import DATASETS

    columns = list(DATASETS[DATASET].schema.names)
    if store is None:
        from research_warehouse.config import get_research_store_dir
        from research_warehouse.store import ResearchStore

        root = get_research_store_dir()
        store = ResearchStore(root) if root is not None else None
    elif not Path(store.root).exists():
        # An unreachable lake must fail loudly, never read as "no regimes".
        raise FileNotFoundError(f"research lake not found at {store.root}")
    rows = store.read_rows(DATASET, symbols=[str(symbol).upper()]) if store is not None else []
    if rows and rule_version is None:
        present = {str(row.get("rule_version")) for row in rows}
        rule_version = max(present, key=lambda version: (_rank(version), version))
    rows = [row for row in rows if str(row.get("rule_version")) == str(rule_version)]
    first: dict[date, dict] = {}
    for row in sorted(rows, key=lambda r: r.get("computed_at") or datetime.min.replace(tzinfo=timezone.utc)):
        day = _as_day(row.get("session_date"))
        first.setdefault(day, {**row, "session_date": day, "next_session_date": _as_day(row.get("next_session_date"))})
    frame = pd.DataFrame([first[day] for day in sorted(first)], columns=columns)
    if as_of == "next_open":
        frame["trade_session_date"] = frame["next_session_date"]
    elif as_of != "close":
        raise ValueError("as_of is 'close' or 'next_open'")
    return frame


__all__ = [
    "AXES",
    "DATASET",
    "RULE_HISTORY",
    "RULE_VERSION",
    "STRUCTURAL_RULE_VERSION",
    "build_rows",
    "read_regimes",
    "run_build",
    "smooth_labels",
    "structural_raw",
]
