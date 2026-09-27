"""The backtester's setup registry: long and short daily setups as point-in-time rules. Shadow only.

A setup is a key, a side, a family, a version and a rule. A rule reads one name's
`Ctx` (its completed daily bars plus cached causal indicators) and returns a boolean
array over those bars plus feature arrays: element ``i`` may read bars ``<= i`` only
(the point-in-time test truncates every name at every flag and demands the same
answer). ``approx`` marks a rule that approximates a live scanner setup from bars.

Adding a setup = one function + one `Setup(...)` line in `REGISTRY`. Bump ``version``
whenever a rule's meaning changes: a run's manifest records every version it used.

Nothing here feeds a live score, alert, tier, gate or Focus list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Mapping, Sequence

import numpy as np

LONG, SHORT = "long", "short"
ATR_BARS = 14
#: The earnings AVWAP window (`long_setups.AVWAPE_SESSIONS`): reaction 2-120 sessions back.
AVWAPE_SESSIONS = (2, 120)
FAV_ZONE_SESSIONS = (2, 63)
STRENGTH_MIN_SMA50_ATR = 2.0
HIGH_52W_BARS = 252
#: Bars handed to the live leader-pullback function (long_lab.LIVE_WINDOW_BARS).
LIVE_WINDOW_BARS = 400
SLACK = 1e-6  # prefilters are supersets: floating sums differ in the last bits
#: strong_deep_pullback (p11 study, 2026-09-27): RS share floor and depth band off the 60-session high.
DEEP_PULLBACK_RS_MIN = 0.9
DEEP_PULLBACK_DEPTH = (0.12, 0.30)
DEEP_PULLBACK_HIGH_BARS = 60
#: earnings_miss_short: gap of at most -1 ATR(14) on a miss of at least 5%, the surprise dated
#: within this many calendar days before the reaction bar.
MISS_GAP_MAX_ATR = -1.0
MISS_SURPRISE_MAX_PCT = -5.0
SURPRISE_MATCH_DAYS = 4
#: laggard_thrust / weakest_near_60d_high: the two grid-search themes picked on 2018-2023 (p11).
LAGGARD_RS_MAX = 0.2
LAGGARD_RET20_MIN = 0.10
LAGGARD_VOLUME_RATIO_MIN = 1.3
WEAKEST_RS_MAX = 0.1
WEAKEST_DEPTH_MAX = 0.05


# ---------------------------------------------------------------- causal helpers
def rolling(values: np.ndarray, window: int, how: Callable) -> np.ndarray:
    """``how`` over each full trailing window (bars <= i); NaN until ``window`` bars exist."""
    out = np.full(len(values), np.nan)
    if window > 0 and len(values) >= window:
        view = np.lib.stride_tricks.sliding_window_view(values, window)
        with np.errstate(all="ignore"):
            out[window - 1:] = how(view, axis=1)
    return out


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """EMA seeded by the SMA of the first ``span`` bars (long_setups.ema); NaN before. Causal."""
    out = np.full(len(values), np.nan)
    if len(values) < span:
        return out
    alpha = 2.0 / (span + 1.0)
    level = float(np.mean(values[:span]))
    out[span - 1] = level
    vals = values.tolist()
    for i in range(span, len(vals)):
        level += alpha * (vals[i] - level)
        out[i] = level
    return out


def shift(values: np.ndarray, k: int) -> np.ndarray:
    """``values[i - k]`` at ``i`` (NaN before); k > 0 looks back only."""
    out = np.full(len(values), np.nan)
    if 0 < k < len(values):
        out[k:] = values[:-k]
    return out


def finite(*arrays: np.ndarray) -> np.ndarray:
    ok = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        ok &= np.isfinite(arr)
    return ok


# ---------------------------------------------------------------- one name's bars
@dataclass
class Ctx:
    """One name's completed daily bars (oldest first) and the day's cross-section facts.

    ``rs_share`` = share of the day's names its 63-session return beats (NaN when unknown),
    ``rs_decile`` = 1-10 (NaN when unknown); both are computed from bars <= that day.
    """

    symbol: str
    dates: np.ndarray  # datetime64[D]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    earnings: Sequence[date] = ()
    rs_share: np.ndarray | None = None
    rs_decile: np.ndarray | None = None
    #: EPS surprise % per earnings date (known at the report); empty when unknown.
    surprises: Mapping[date, float] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.close)
        if self.rs_share is None:
            self.rs_share = np.full(n, np.nan)
        if self.rs_decile is None:
            self.rs_decile = np.full(n, np.nan)

    def __len__(self) -> int:
        return len(self.close)

    def truncated(self, end: int) -> "Ctx":
        """The same name with only bars ``<= end`` (the point-in-time test's view)."""
        k = end + 1
        return Ctx(self.symbol, self.dates[:k], self.open[:k], self.high[:k], self.low[:k],
                   self.close[:k], self.volume[:k], self.earnings,
                   self.rs_share[:k].copy(), self.rs_decile[:k].copy(), surprises=self.surprises)

    def _memo(self, key: str, build: Callable[[], Any]) -> Any:
        if key not in self.cache:
            self.cache[key] = build()
        return self.cache[key]

    def sma(self, n: int) -> np.ndarray:
        return self._memo(f"sma{n}", lambda: rolling(self.close, n, np.mean))

    def ema(self, n: int) -> np.ndarray:
        return self._memo(f"ema{n}", lambda: ema(self.close, n))

    @property
    def prev_close(self) -> np.ndarray:
        return self._memo("prev_close", lambda: shift(self.close, 1))

    @property
    def atr(self) -> np.ndarray:
        """ATR(14): the mean true range of the last 14 bars (long_lab.Series.atr)."""
        def build():
            prev = self.prev_close
            tr = np.where(np.isnan(prev), self.high - self.low,
                          np.maximum(self.high, prev) - np.minimum(self.low, prev))
            return rolling(tr, ATR_BARS, np.mean)
        return self._memo("atr", build)

    @property
    def atr20(self) -> np.ndarray:
        """The scan's atr20 (long_lab.scan_atr20): mean TR of 20 bars, the first counted high - low."""
        def build():
            hl = self.high - self.low
            prev = self.prev_close
            tr = np.where(np.isnan(prev), hl, np.maximum(hl, np.maximum(np.abs(self.high - prev),
                                                                         np.abs(self.low - prev))))
            out = rolling(tr, 20, np.mean)
            if len(hl) >= 20:
                out[19:] += (hl[: len(hl) - 19] - tr[: len(tr) - 19]) / 20.0
            return np.where(out > 0, out, np.nan)
        return self._memo("atr20", build)

    def max_high(self, n: int) -> np.ndarray:
        return self._memo(f"maxh{n}", lambda: rolling(self.high, n, np.max))

    def min_low(self, n: int) -> np.ndarray:
        return self._memo(f"minl{n}", lambda: rolling(self.low, n, np.min))

    def prior_max_high(self, n: int) -> np.ndarray:
        """Max high of the ``n`` bars before ``i`` (excludes ``i``)."""
        return self._memo(f"pmaxh{n}", lambda: shift(self.max_high(n), 1))

    def prior_min_low(self, n: int) -> np.ndarray:
        return self._memo(f"pminl{n}", lambda: shift(self.min_low(n), 1))

    def ret(self, n: int) -> np.ndarray:
        return self._memo(f"ret{n}", lambda: self.close / shift(self.close, n) - 1.0)

    # -- earnings
    @property
    def reactions(self) -> np.ndarray:
        """Earnings reaction bars (long_lab.reaction_day): of the first session on/after each
        date and the next, the bigger |open - prior close| (tie: the first). Sorted, unique."""
        def build():
            n = len(self)
            if not len(self.earnings) or n < 3:
                return np.zeros(0, dtype=np.int64)
            days = np.array(sorted({np.datetime64(d, "D") for d in self.earnings}))
            first = np.searchsorted(self.dates, days, side="left")
            first = first[(first >= 1) & (first + 1 < n)]
            if not len(first):
                return np.zeros(0, dtype=np.int64)
            gap_a = np.abs(self.open[first] - self.close[first - 1])
            gap_b = np.abs(self.open[first + 1] - self.close[first])
            return np.unique(np.where(gap_b > gap_a, first + 1, first)).astype(np.int64)
        return self._memo("reactions", build)

    def latest_reaction(self) -> np.ndarray:
        """At each bar ``i`` the latest reaction ``r`` with ``r + 1 <= i`` (-1 when none)."""
        def build():
            n = len(self)
            react = self.reactions
            pos = np.searchsorted(react, np.arange(n) - 1, side="right") - 1
            return np.where(pos >= 0, react[np.clip(pos, 0, None)] if len(react) else -1, -1)
        return self._memo("latest_reaction", build)

    def earnings_avwap(self, sessions: tuple[int, int] = AVWAPE_SESSIONS) -> dict[str, np.ndarray]:
        """The study-anchor earnings AVWAP (`long_setups.earnings_avwap`): anchored the session
        before the latest reaction when that reaction is ``sessions`` back; OHLC/4 price and the
        running-deviation sigma. ``{level, sigma, since, reaction}``, NaN / -1 when unknown."""
        return self._memo(f"avwape{sessions}", lambda: self._earnings_avwap(sessions))

    def _earnings_avwap(self, sessions: tuple[int, int]) -> dict[str, np.ndarray]:
        n = len(self)
        level, sigma = np.full(n, np.nan), np.full(n, np.nan)
        latest = self.latest_reaction()
        idx = np.arange(n)
        since = np.where(latest >= 0, idx - latest, -1)
        valid = (latest >= 1) & (since >= sessions[0]) & (since <= sessions[1])
        tp = (self.open + self.high + self.low + self.close) / 4.0
        vol = np.where(np.isfinite(self.volume) & (self.volume > 0), self.volume, 0.0)
        for r in np.unique(latest[valid]):
            anchor = int(r) - 1
            stop = min(n, int(r) + sessions[1] + 1)
            seg_v, seg_p = vol[anchor:stop], tp[anchor:stop]
            cum_v = np.cumsum(seg_v)
            cum_vp = np.cumsum(seg_p * seg_v)
            with np.errstate(all="ignore"):
                running = np.where(cum_v > 0, cum_vp / np.where(cum_v > 0, cum_v, 1.0), np.nan)
                dev = np.where(seg_v > 0, seg_p - running, 0.0)
                cum_sd = np.cumsum(np.nan_to_num(dev * dev * seg_v))
                lv = np.where(cum_v > 0, cum_vp / np.where(cum_v > 0, cum_v, 1.0), np.nan)
                sg = np.where(cum_v > 0, np.sqrt(cum_sd / np.where(cum_v > 0, cum_v, 1.0)), np.nan)
            rows = np.nonzero(valid & (latest == r))[0]
            level[rows] = lv[rows - anchor]
            sigma[rows] = sg[rows - anchor]
        return {"level": level, "sigma": sigma, "since": np.where(valid, since, -1),
                "reaction": np.where(valid, latest, -1)}

    def gap_atr(self) -> np.ndarray:
        """(open - prior close) / prior ATR(14) at each bar (long_lab.gap_atr)."""
        def build():
            prior_atr = shift(self.atr, 1)
            with np.errstate(all="ignore"):
                return np.where(prior_atr > 0, (self.open - self.prev_close) / prior_atr, np.nan)
        return self._memo("gap_atr", build)

    # -- the live rules' bar list (built once per name)
    def live_bars(self, i: int) -> list[dict[str, Any]]:
        if "live_bars" not in self.cache:
            days = [str(d) for d in self.dates.astype("datetime64[D]")]
            vols = [float(v) if math.isfinite(v) else None for v in self.volume.tolist()]
            self.cache["live_bars"] = [
                {"date": days[j], "open": o, "high": h, "low": lo, "close": c, "volume": vols[j]}
                for j, (o, h, lo, c) in enumerate(zip(self.open.tolist(), self.high.tolist(),
                                                      self.low.tolist(), self.close.tolist(), strict=True))]
        return self.cache["live_bars"][max(0, i - LIVE_WINDOW_BARS + 1):i + 1]


Result = tuple[np.ndarray, dict[str, np.ndarray]]


def _none(ctx: Ctx) -> Result:
    return np.zeros(len(ctx), dtype=bool), {}


# ---------------------------------------------------------------- long rules
def leader_pullback_live(ctx: Ctx) -> Result:
    """The live `long_setups.leader_pullback` on bars up to each close (last 400) with the day's
    RS share. A vectorized superset prefilter (100/200-day, 8-25% off the 60-session high, at the
    VWAP from the high or near the 21/50 EMA, lighter pullback volume) then the live function
    confirms each survivor, so the answer is the live one. No sector / top-pattern facts."""
    import long_setups as ls

    n = len(ctx)
    mask = np.zeros(n, dtype=bool)
    feats = {k: np.full(n, np.nan) for k in ("pct_off_high", "pct_under_avwap", "strength")}
    if n < ls.TREND_SMA:
        return mask, feats
    c, h, lo, v = ctx.close, ctx.high, ctx.low, ctx.volume
    atr = ctx.atr20
    pre = np.arange(n) >= ls.TREND_SMA - 1
    pre &= np.isfinite(atr)
    for length in ls.TREND_SMAS:
        pre &= c > ctx.sma(length) * (1 - SLACK)
    look = ls.SWING_HIGH_LOOKBACK
    win = np.lib.stride_tricks.sliding_window_view(h, look)
    # latest index of the max in the window (ties -> the latest bar, as the live `max` key)
    hi_idx = np.full(n, -1)
    hi_idx[look - 1:] = np.arange(look - 1, n) - np.argmax(win[:, ::-1], axis=1)
    pre &= (hi_idx >= 0) & (hi_idx < np.arange(n))
    swing = np.where(hi_idx >= 0, h[np.clip(hi_idx, 0, None)], np.nan)
    with np.errstate(all="ignore"):
        off = (swing - c) / swing * 100.0
    pre &= (off >= ls.OFF_HIGH_PCT[0] - SLACK) & (off <= ls.OFF_HIGH_PCT[1] + SLACK)
    vol0 = np.where(np.isfinite(v), v, 0.0)
    cpv = np.concatenate(([0.0], np.cumsum((h + lo + c) / 3.0 * vol0)))
    cv = np.concatenate(([0.0], np.cumsum(vol0)))
    safe = np.clip(hi_idx, 0, None)
    idx = np.arange(n)
    with np.errstate(all="ignore"):
        vwap = (cpv[idx + 1] - cpv[safe]) / (cv[idx + 1] - cv[safe])
        under = (vwap - c) / vwap * 100.0
    at_vwap = (under >= ls.UNDER_AVWAP_PCT[0] - 1e-4) & (under <= ls.UNDER_AVWAP_PCT[1] + 1e-4)
    near = np.zeros(n, dtype=bool)
    for length in ls.EMA_LENGTHS:
        near |= np.abs(c - ctx.ema(length)) <= ls.EMA_NEAR_ATR * atr * 1.05 + 1e-9
    pre &= at_vwap | near
    run_start = np.clip(safe - ls.RUN_VOLUME_SESSIONS + 1, 0, None)
    with np.errstate(all="ignore"):
        run_mean = (cv[safe + 1] - cv[run_start]) / (safe + 1 - run_start)
        pb_mean = (cv[idx + 1] - cv[safe + 1]) / np.maximum(idx - safe, 1)
    pre &= (run_mean > 0) & (pb_mean <= run_mean * (1 + SLACK))
    eligible = ctx.cache.get("eligible")  # the replay's causal price / volume / date gate
    if eligible is not None:
        pre &= eligible
    rs = ctx.rs_share
    for i in np.nonzero(pre)[0]:
        share = float(rs[i]) if math.isfinite(rs[i]) else None
        row = ls.leader_pullback(ctx.live_bars(int(i)), atr=float(atr[i]), rs_percentile=share)
        if row is None:
            continue
        mask[i] = True
        for key in feats:
            value = row.get(key)
            feats[key][i] = float(value) if value is not None else np.nan
    return mask, feats


def strength_under_avwape(ctx: Ctx) -> Result:
    """The promotion tier merged 2026-09-27: a live leader pullback that is strong - (close -
    SMA50) / ATR20 >= 2 - AND closed under the study-anchor earnings AVWAP. The live tier also
    wants SPY above a rising 20-day; here that is left to the regime axes (paint it there)."""
    base, feats = ctx._memo("lp_live", lambda: leader_pullback_live(ctx))
    with np.errstate(all="ignore"):
        dist = (ctx.close - ctx.sma(50)) / ctx.atr20
    av = ctx.earnings_avwap(AVWAPE_SESSIONS)
    mask = base & (dist >= STRENGTH_MIN_SMA50_ATR) & np.isfinite(av["level"]) & (av["sigma"] > 0) \
        & (ctx.close < av["level"])
    with np.errstate(all="ignore"):
        z = (ctx.close - av["level"]) / av["sigma"]
    return mask, {**feats, "strength_sma50_atr": dist, "avwape_z": z,
                  "sessions_since_earnings": av["since"].astype(float)}


def _leader_pullback_cached(ctx: Ctx) -> Result:
    return ctx._memo("lp_live", lambda: leader_pullback_live(ctx))


_leader_pullback_cached.__doc__ = leader_pullback_live.__doc__


def _fav_zone(ctx: Ctx, side: str) -> Result:
    av = ctx.earnings_avwap(FAV_ZONE_SESSIONS)
    level, sigma, react = av["level"], av["sigma"], av["reaction"]
    ok = np.isfinite(level) & np.isfinite(sigma) & (react >= 1)
    safe = np.clip(react, 1, None)
    c = ctx.close
    if side == LONG:
        mask = ok & (level <= c) & (c <= level + sigma)
    else:
        down = ctx.open[safe] < ctx.close[safe - 1]
        mask = ok & down & (level - sigma <= c) & (c <= level)
    with np.errstate(all="ignore"):
        z = (c - level) / sigma
    return mask, {"sessions_since_earnings": av["since"].astype(float), "avwape_z": z}


def favourite_zone_long(ctx: Ctx) -> Result:
    """Retired favourite-zone long (contrast, long_lab rule c): close between the AVWAP anchored
    the session before the latest earnings reaction and its +1 sigma band, 2-63 sessions on."""
    return _fav_zone(ctx, LONG)


def high_52w_breakout(ctx: Ctx) -> Result:
    """Close above the highest high of the prior 252 sessions."""
    prior = ctx.prior_max_high(HIGH_52W_BARS)
    mask = finite(prior) & (ctx.close > prior)
    with np.errstate(all="ignore"):
        return mask, {"pct_over_52w_high": (ctx.close / prior - 1.0) * 100.0}


def post_earnings_drift_live(ctx: Ctx) -> Result:
    """The live `long_setups.post_earnings_drift` rule from bars: an earnings reaction that gapped
    up >= 1 ATR(14) and closed in the top half of its range, now 4-7 sessions on, no close at or
    under the gap-day low since, close over the VWAP (HLC/3) anchored at the gap day."""
    import long_setups as ls

    n = len(ctx)
    mask = np.zeros(n, dtype=bool)
    gap_atr_f, after_f = np.full(n, np.nan), np.full(n, np.nan)
    gaps = ctx.gap_atr()
    atr20 = ctx.atr20
    o, h, lo, c, v = ctx.open, ctx.high, ctx.low, ctx.close, ctx.volume
    for g in ctx.reactions:
        g = int(g)
        size = gaps[g]
        if not (math.isfinite(size) and size >= ls.PED_GAP_MIN_ATR and o[g] > c[g - 1]):
            continue
        rng = h[g] - lo[g]
        if rng <= 0 or (c[g] - lo[g]) / rng < ls.PED_CLOSE_LOCATION_MIN:
            continue
        vw_num = vw_den = 0.0
        broken = False
        for i in range(g, min(n, g + ls.PED_SESSIONS[1] + 1)):
            vol = v[i]
            if not math.isfinite(vol) or vol < 0:
                break
            vw_num += (h[i] + lo[i] + c[i]) / 3.0 * vol
            vw_den += vol
            if i > g and c[i] <= lo[g]:
                broken = True
            if broken:
                break
            after = i - g
            if after < ls.PED_SESSIONS[0] or not math.isfinite(atr20[i]):
                continue
            if vw_den > 0 and c[i] > vw_num / vw_den:
                mask[i] = True
                gap_atr_f[i], after_f[i] = size, after
    return mask, {"gap_atr": gap_atr_f, "sessions_after_gap": after_f}


def strong_deep_pullback(ctx: Ctx) -> Result:
    """p11 study: a 63-day RS top-decile name (share >= 0.9) closing over its 200-day SMA and
    12-30% under its highest high of the last 60 sessions. Leaders that correct hard revert up."""
    high = ctx.max_high(DEEP_PULLBACK_HIGH_BARS)
    with np.errstate(all="ignore"):
        depth = 1.0 - ctx.close / high
    s200 = ctx.sma(200)
    mask = finite(high, s200) & (ctx.rs_share >= DEEP_PULLBACK_RS_MIN) & (ctx.close > s200) \
        & (depth >= DEEP_PULLBACK_DEPTH[0]) & (depth < DEEP_PULLBACK_DEPTH[1])
    return mask, {"pct_off_60d_high": depth * 100.0}


def _volume_ratio(ctx: Ctx) -> np.ndarray:
    """Mean volume of the last 5 sessions over the last 50 (unknown when any volume is missing)."""
    def build():
        with np.errstate(all="ignore"):
            return rolling(ctx.volume, 5, np.mean) / rolling(ctx.volume, 50, np.mean)
    return ctx._memo("vr5_50", build)


def laggard_thrust(ctx: Ctx) -> Result:
    """p11 search theme: a bottom-quintile 63-day RS name (share < 0.2) back over its 100- and
    200-day SMAs, up 10%+ in 20 sessions on 5-day volume >= 1.3x its 50-day mean."""
    s100, s200 = ctx.sma(100), ctx.sma(200)
    ret20, vr = ctx.ret(20), _volume_ratio(ctx)
    mask = finite(s100, s200, ret20, vr) & (ctx.close > s100) & (ctx.close > s200) \
        & (ctx.rs_share < LAGGARD_RS_MAX) & (ret20 >= LAGGARD_RET20_MIN) & (vr >= LAGGARD_VOLUME_RATIO_MIN)
    return mask, {"ret20_pct": ret20 * 100.0, "volume_ratio": vr}


def rising_20_50_baseline(ctx: Ctx) -> Result:
    """Baseline (long_lab rule d): close above a rising 20-day and a rising 50-day SMA
    (rising = above its value 5 sessions earlier)."""
    s20, s50 = ctx.sma(20), ctx.sma(50)
    p20, p50 = shift(s20, 5), shift(s50, 5)
    c = ctx.close
    return finite(s20, s50, p20, p50) & (c > s20) & (c > s50) & (s20 > p20) & (s50 > p50), {}


# ---------------------------------------------------------------- short rules
def favourite_zone_short(ctx: Ctx) -> Result:
    """Favourite-zone short: after an earnings gap DOWN, close between the earnings AVWAP (anchored
    the session before the reaction) and its -1 sigma band, 2-63 sessions on."""
    return _fav_zone(ctx, SHORT)


def weak_rally_to_avwape(ctx: Ctx) -> Result:
    """Weak name rallies up to the earnings AVWAP from below: bottom-decile 63-day RS, close under
    the 100- and 200-day, the day's high tags the AVWAP and the close stays under it, the prior
    close under it too."""
    av = ctx.earnings_avwap(AVWAPE_SESSIONS)
    level = av["level"]
    c = ctx.close
    mask = np.isfinite(level) & (ctx.rs_decile == 1) & finite(ctx.sma(100), ctx.sma(200)) \
        & (c < ctx.sma(100)) & (c < ctx.sma(200)) & (ctx.high >= level) & (c < level) \
        & (ctx.prev_close < level)
    with np.errstate(all="ignore"):
        z = (c - level) / av["sigma"]
    return mask, {"avwape_z": z, "sessions_since_earnings": av["since"].astype(float)}


def low_52w_breakdown(ctx: Ctx) -> Result:
    """Close under the lowest low of the prior 252 sessions."""
    prior = ctx.prior_min_low(HIGH_52W_BARS)
    mask = finite(prior) & (ctx.close < prior)
    with np.errstate(all="ignore"):
        return mask, {"pct_under_52w_low": (1.0 - ctx.close / prior) * 100.0}


def earnings_miss_short(ctx: Ctx) -> Result:
    """p11 study: the session after an earnings reaction that gapped DOWN >= 1 ATR(14) on an EPS
    miss of 5%+ (surprise dated 0-4 days before the reaction bar), closing under its 20-day SMA.
    The surprise is known at the report, before the reaction bar opens."""
    n = len(ctx)
    mask = np.zeros(n, dtype=bool)
    surprise_f, gap_f = np.full(n, np.nan), np.full(n, np.nan)
    known = sorted((np.datetime64(d, "D"), float(v)) for d, v in (ctx.surprises or {}).items()
                   if v is not None and math.isfinite(float(v)))
    if not known or not len(ctx.reactions):
        return mask, {"surprise_pct": surprise_f, "gap_atr": gap_f}
    days = np.array([d for d, _ in known], dtype="datetime64[D]")
    vals = np.array([v for _, v in known])
    gaps, s20 = ctx.gap_atr(), ctx.sma(20)
    for r in ctx.reactions.tolist():
        i = r + 1
        if i >= n:
            continue
        pos = int(np.searchsorted(days, ctx.dates[r], side="right")) - 1
        if pos < 0 or (ctx.dates[r] - days[pos]).astype(int) > SURPRISE_MATCH_DAYS:
            continue
        gap, surprise = gaps[r], vals[pos]
        if not (math.isfinite(gap) and gap <= MISS_GAP_MAX_ATR and surprise <= MISS_SURPRISE_MAX_PCT):
            continue
        if math.isfinite(s20[i]) and ctx.close[i] < s20[i]:
            mask[i] = True
            surprise_f[i], gap_f[i] = surprise, gap
    return mask, {"surprise_pct": surprise_f, "gap_atr": gap_f}


def weakest_near_60d_high(ctx: Ctx) -> Result:
    """p11 search theme: a bottom-decile 63-day RS name (share < 0.1) under its 100- and 200-day
    SMAs that has rallied to within 5% of its highest high of the last 60 sessions."""
    s100, s200 = ctx.sma(100), ctx.sma(200)
    high = ctx.max_high(DEEP_PULLBACK_HIGH_BARS)
    with np.errstate(all="ignore"):
        depth = 1.0 - ctx.close / high
    mask = finite(s100, s200, high) & (ctx.close < s100) & (ctx.close < s200) \
        & (ctx.rs_share < WEAKEST_RS_MAX) & (depth < WEAKEST_DEPTH_MAX)
    return mask, {"pct_off_60d_high": depth * 100.0}


def falling_20_50_baseline(ctx: Ctx) -> Result:
    """Baseline: close below a falling 20-day and a falling 50-day SMA."""
    s20, s50 = ctx.sma(20), ctx.sma(50)
    p20, p50 = shift(s20, 5), shift(s50, 5)
    c = ctx.close
    return finite(s20, s50, p20, p50) & (c < s20) & (c < s50) & (s20 < p20) & (s50 < p50), {}


# ---------------------------------------------------------------- the registry
@dataclass(frozen=True)
class Setup:
    key: str
    side: str
    family: str
    version: str
    fn: Callable[[Ctx], Result]
    approx: bool = False
    needs_earnings: bool = False  # reads earnings dates: no dates = no measurement, not a result
    note: str = ""

    def meta(self) -> dict[str, Any]:
        return {"key": self.key, "side": self.side, "family": self.family, "version": self.version,
                "approx": self.approx, "needs_earnings": self.needs_earnings, "note": self.note, "rule": (self.fn.__doc__ or "").strip()}


REGISTRY: tuple[Setup, ...] = (
    Setup("leader_pullback", LONG, "pullback", "1", _leader_pullback_cached, approx=True,
          note="live long_setups.leader_pullback; no sector / top-pattern bonus facts in bars"),
    Setup("strength_under_avwape", LONG, "pullback", "1", strength_under_avwape, approx=True, needs_earnings=True,
          note="live promotion tier; its SPY-above-rising-20d part is a regime axis here"),
    Setup("favourite_zone_long", LONG, "earnings_avwap", "1", favourite_zone_long, approx=True, needs_earnings=True,
          note="retired live setup, kept for contrast; anchor from the earnings-dates store"),
    Setup("high_52w_breakout", LONG, "breakout", "1", high_52w_breakout),
    Setup("post_earnings_drift", LONG, "earnings_drift", "1", post_earnings_drift_live, approx=True, needs_earnings=True,
          note="live long_setups.post_earnings_drift; gap size in ATR(14) of the prior bar"),
    Setup("rising_20_50_baseline", LONG, "baseline", "1", rising_20_50_baseline),
    Setup("strong_deep_pullback", LONG, "pullback", "1", strong_deep_pullback,
          note="p11 study 2026-09-27: RS top decile over the 200d, 12-30% off the 60d high"),
    Setup("laggard_thrust", LONG, "breakout", "1", laggard_thrust,
          note="p11 grid-search theme picked on 2018-2023; thin (about 100 trades a period)"),
    Setup("favourite_zone_short", SHORT, "earnings_avwap", "1", favourite_zone_short, approx=True, needs_earnings=True,
          note="mirror of the favourite-zone long after an earnings gap down"),
    Setup("weak_rally_to_avwape", SHORT, "earnings_avwap", "1", weak_rally_to_avwape,
          needs_earnings=True),
    Setup("low_52w_breakdown", SHORT, "breakout", "1", low_52w_breakdown),
    Setup("falling_20_50_baseline", SHORT, "baseline", "1", falling_20_50_baseline),
    Setup("earnings_miss_short", SHORT, "earnings_drift", "1", earnings_miss_short, needs_earnings=True,
          note="p11 study 2026-09-27: day after a >= 1 ATR gap down on a 5%+ EPS miss, under the 20d"),
    Setup("weakest_near_60d_high", SHORT, "counter_trend", "1", weakest_near_60d_high,
          note="p11 grid-search theme picked on 2018-2023; train carried by 2020"),
)


def by_key(keys: Sequence[str] | None = None, registry: Sequence[Setup] = REGISTRY) -> list[Setup]:
    """The named setups in registry order (all when ``keys`` is empty); unknown keys raise."""
    if not keys:
        return list(registry)
    known = {s.key: s for s in registry}
    missing = [k for k in keys if k not in known]
    if missing:
        raise KeyError(f"unknown setup(s): {', '.join(missing)}; known: {', '.join(known)}")
    return [s for s in registry if s.key in set(keys)]


def evaluate(setup: Setup, ctx: Ctx) -> Result:
    mask, feats = setup.fn(ctx)
    return np.asarray(mask, dtype=bool), {k: np.asarray(v, dtype=float) for k, v in (feats or {}).items()}


__all__ = ["Ctx", "LONG", "REGISTRY", "SHORT", "Setup", "by_key", "evaluate"]

