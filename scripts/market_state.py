"""Pure SPY market-state / pullback-episode engine (plan.md Packet B, sec 16, 24).

No I/O, no Qt, no broker calls: the engine consumes complete 5-minute bars in
order and produces typed state snapshots plus append-only pullback episodes.
Long and short logic share one signed coordinate system (`side_sign`): +1 for
the bullish direction, -1 for the bearish direction. Only labels branch by
side, so mirrored input must produce mirrored decisions (tested).

Thresholds live in a versioned MarketStateConfig (initial values are the
plan's section-24 shadow-test hypotheses, not tuned truth). Incomplete or
stale bars can never transition state - they only mark the snapshot stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


ENGINE_VERSION = "spy_state_v1"


class MarketState(str, Enum):
    PREOPEN = "PREOPEN"
    OPENING_DISCOVERY = "OPENING_DISCOVERY"
    BULL_IMPULSE = "BULL_IMPULSE"
    BEAR_IMPULSE = "BEAR_IMPULSE"
    RANGE = "RANGE"
    COUNTERMOVE_ARMED = "COUNTERMOVE_ARMED"
    COUNTERMOVE_ACTIVE = "COUNTERMOVE_ACTIVE"
    STABILIZING = "STABILIZING"
    TREND_RESUMED = "TREND_RESUMED"
    REGIME_FAILED = "REGIME_FAILED"


@dataclass(frozen=True)
class M5Bar:
    """One five-minute bar; `ts` is the bar's completion time."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    complete: bool = True


@dataclass(frozen=True)
class MarketStateConfig:
    version: str = ENGINE_VERSION
    # --- opening / warmup ---
    opening_bars: int = 3
    atr_window: int = 20
    min_atr_bars: int = 5
    # --- strong-trend entry/exit (sec 24) ---
    min_day_return_pct: float = 0.35
    day_return_atr_fraction: float = 0.25
    vwap_side_window: int = 12
    vwap_side_min_aligned: int = 8
    trend_enter_score: float = 0.65
    trend_enter_bars: int = 2
    trend_exit_score: float = 0.35
    trend_exit_bars: int = 2
    # --- counter-move start (sec 24) ---
    countermove_min_pct: float = 0.12
    countermove_atr_fraction: float = 0.30
    min_impulse_atr: float = 0.60
    arm_countertrend_closes: int = 2  # of the last 3
    # --- resumption / failure (sec 24) ---
    min_episode_bars: int = 2
    resume_recovery_fraction: float = 0.35
    fail_vwap_atr: float = 0.25
    fail_vwap_closes: int = 2
    fail_depth_atr: float = 1.5
    fail_max_bars: int = 12
    # --- data freshness ---
    max_bar_gap: timedelta = timedelta(minutes=10)


@dataclass
class EpisodeEvent:
    state: MarketState
    ts: datetime
    price: float
    depth_atr: float


@dataclass
class PullbackEpisode:
    """One counter-move against an active impulse; append-only events."""

    episode_id: str
    engine_version: str
    side_sign: int  # +1 pullback in a bull trend, -1 bounce in a bear trend
    impulse_extreme: float
    armed_ts: datetime
    events: list[EpisodeEvent] = field(default_factory=list)
    outcome: str = ""  # RESUMED / FAILED / "" while open

    @property
    def direction(self) -> str:
        return "BULL_PULLBACK" if self.side_sign > 0 else "BEAR_BOUNCE"

    def record(self, state: MarketState, ts: datetime, price: float, depth_atr: float) -> None:
        self.events.append(EpisodeEvent(state, ts, price, depth_atr))


@dataclass(frozen=True)
class MarketStateSnapshot:
    state: MarketState
    ts: datetime | None
    side_sign: int  # trend direction while one is active, else 0
    trend_score: float
    day_return_pct: float
    vwap: float | None
    m5_atr: float | None
    countermove_depth_atr: float
    stale: bool
    engine_version: str = ENGINE_VERSION


class MarketStateEngine:
    """Feed complete 5-minute bars in order; read typed snapshots/episodes."""

    def __init__(
        self,
        prior_close: float,
        config: MarketStateConfig | None = None,
        daily_atr_pct: float | None = None,
    ) -> None:
        self.config = config or MarketStateConfig()
        self.prior_close = float(prior_close)
        self.daily_atr_pct = daily_atr_pct
        self.state = MarketState.PREOPEN
        self.side_sign = 0
        self.bars: list[M5Bar] = []
        self.episodes: list[PullbackEpisode] = []
        self._cum_vol = 0.0
        self._cum_tpv = 0.0
        self._vwap: float | None = None
        self._vwap_side_history: list[int] = []  # +1 close above vwap, -1 below
        self._trend_enter_streak = 0
        self._trend_exit_streak = 0
        self._impulse_extreme: float | None = None
        self._impulse_start_price: float | None = None
        self._countermove_extreme: float | None = None
        self._stabilize_pivot: float | None = None
        self._resume_cooldown = 0
        self._last_snapshot = MarketStateSnapshot(
            state=self.state,
            ts=None,
            side_sign=0,
            trend_score=0.0,
            day_return_pct=0.0,
            vwap=None,
            m5_atr=None,
            countermove_depth_atr=0.0,
            stale=False,
        )

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------
    @property
    def snapshot(self) -> MarketStateSnapshot:
        return self._last_snapshot

    @property
    def active_episode(self) -> PullbackEpisode | None:
        if self.episodes and not self.episodes[-1].outcome:
            return self.episodes[-1]
        return None

    def on_bar(self, bar: M5Bar) -> MarketStateSnapshot:
        if not bar.complete or self._is_stale(bar):
            # Incomplete or gapped data can never transition state (sec 16.3).
            self._last_snapshot = self._make_snapshot(bar.ts, stale=True)
            return self._last_snapshot

        self.bars.append(bar)
        if bar.volume > 0:
            tp = (bar.high + bar.low + bar.close) / 3.0
            self._cum_vol += bar.volume
            self._cum_tpv += tp * bar.volume
            self._vwap = self._cum_tpv / self._cum_vol
        if self._vwap is not None:
            self._vwap_side_history.append(1 if bar.close >= self._vwap else -1)

        if len(self.bars) <= self.config.opening_bars:
            self.state = MarketState.OPENING_DISCOVERY
            self._last_snapshot = self._make_snapshot(bar.ts, stale=False)
            return self._last_snapshot

        self._step(bar)
        self._last_snapshot = self._make_snapshot(bar.ts, stale=False)
        return self._last_snapshot

    # ------------------------------------------------------------------
    # features
    # ------------------------------------------------------------------
    def _is_stale(self, bar: M5Bar) -> bool:
        if not self.bars:
            return False
        return (bar.ts - self.bars[-1].ts) > self.config.max_bar_gap

    def _day_return_pct(self, close: float) -> float:
        if not self.prior_close:
            return 0.0
        return (close - self.prior_close) / self.prior_close * 100.0

    def _m5_atr(self) -> float | None:
        bars = self.bars
        if len(bars) < self.config.min_atr_bars + 1:
            return None
        window = bars[-(self.config.atr_window + 1):]
        trs = []
        for prev, cur in zip(window, window[1:]):
            trs.append(
                max(
                    cur.high - cur.low,
                    abs(cur.high - prev.close),
                    abs(cur.low - prev.close),
                )
            )
        if not trs:
            return None
        return sum(trs) / len(trs)

    def _trend_score(self, side_sign: int, bar: M5Bar) -> float:
        """Fraction of aligned trend conditions met (0..1)."""
        conditions = []
        day_return = self._day_return_pct(bar.close)
        threshold = self.config.min_day_return_pct
        if self.daily_atr_pct:
            threshold = max(threshold, self.config.day_return_atr_fraction * self.daily_atr_pct)
        conditions.append(side_sign * day_return >= threshold)
        if self._vwap is not None:
            conditions.append(side_sign * (bar.close - self._vwap) >= 0)
        window = self._vwap_side_history[-self.config.vwap_side_window:]
        if window:
            aligned = sum(1 for s in window if s == side_sign)
            required = min(self.config.vwap_side_min_aligned, len(window))
            conditions.append(aligned >= required)
        if len(self.bars) >= 4:
            closes = [b.close for b in self.bars[-4:]]
            conditions.append(side_sign * (closes[-1] - closes[0]) > 0)
        if not conditions:
            return 0.0
        return sum(bool(c) for c in conditions) / len(conditions)

    def _countertrend_closes(self, side_sign: int, lookback: int = 3) -> int:
        bars = self.bars[-(lookback + 1):]
        count = 0
        for prev, cur in zip(bars, bars[1:]):
            if side_sign * (cur.close - prev.close) < 0:
                count += 1
        return count

    def _countermove_depth(self) -> float:
        if self._impulse_extreme is None or not self.bars:
            return 0.0
        adverse = self._adverse_extreme_price()
        return self.side_sign * (self._impulse_extreme - adverse)

    def _adverse_extreme_price(self) -> float:
        if self._countermove_extreme is not None:
            return self._countermove_extreme
        bar = self.bars[-1]
        return bar.low if self.side_sign > 0 else bar.high

    # ------------------------------------------------------------------
    # state machine
    # ------------------------------------------------------------------
    def _step(self, bar: M5Bar) -> None:
        if self._resume_cooldown > 0:
            self._resume_cooldown -= 1

        if self.state in (MarketState.OPENING_DISCOVERY, MarketState.RANGE, MarketState.PREOPEN):
            if self.state == MarketState.OPENING_DISCOVERY:
                self.state = MarketState.RANGE
            self._try_enter_impulse(bar)
            return
        if self.state in (MarketState.BULL_IMPULSE, MarketState.BEAR_IMPULSE, MarketState.TREND_RESUMED):
            self._run_impulse(bar)
            return
        if self.state == MarketState.COUNTERMOVE_ARMED:
            self._run_armed(bar)
            return
        if self.state == MarketState.COUNTERMOVE_ACTIVE:
            self._run_active(bar)
            return
        if self.state == MarketState.STABILIZING:
            self._run_stabilizing(bar)
            return
        if self.state == MarketState.REGIME_FAILED:
            # A failed regime falls back to RANGE and may re-qualify later.
            self.state = MarketState.RANGE
            self.side_sign = 0
            self._try_enter_impulse(bar)

    def _try_enter_impulse(self, bar: M5Bar) -> None:
        day_return = self._day_return_pct(bar.close)
        side_sign = 1 if day_return >= 0 else -1
        score = self._trend_score(side_sign, bar)
        if score >= self.config.trend_enter_score and self._m5_atr() is not None:
            self._trend_enter_streak += 1
        else:
            self._trend_enter_streak = 0
        if self._trend_enter_streak >= self.config.trend_enter_bars:
            self.side_sign = side_sign
            self.state = MarketState.BULL_IMPULSE if side_sign > 0 else MarketState.BEAR_IMPULSE
            self._impulse_extreme = bar.high if side_sign > 0 else bar.low
            self._impulse_start_price = bar.close
            self._trend_enter_streak = 0
            self._trend_exit_streak = 0

    def _update_impulse_extreme(self, bar: M5Bar) -> None:
        edge = bar.high if self.side_sign > 0 else bar.low
        if self._impulse_extreme is None or self.side_sign * (edge - self._impulse_extreme) > 0:
            self._impulse_extreme = edge

    def _run_impulse(self, bar: M5Bar) -> None:
        self._update_impulse_extreme(bar)

        score = self._trend_score(self.side_sign, bar)
        if score < self.config.trend_exit_score:
            self._trend_exit_streak += 1
        else:
            self._trend_exit_streak = 0
        if self._trend_exit_streak >= self.config.trend_exit_bars:
            self.state = MarketState.RANGE
            self.side_sign = 0
            self._trend_exit_streak = 0
            return

        if self._resume_cooldown > 0:
            return

        # Arm a counter-move: weakening closes or an adverse EMA8 cross.
        countertrend = self._countertrend_closes(self.side_sign)
        ema8 = self._ema(8)
        adverse_ema_cross = ema8 is not None and self.side_sign * (bar.close - ema8) < 0
        if countertrend >= self.config.arm_countertrend_closes or adverse_ema_cross:
            atr = self._m5_atr()
            impulse_size = (
                abs(self._impulse_extreme - self._impulse_start_price)
                if self._impulse_extreme is not None and self._impulse_start_price is not None
                else 0.0
            )
            if atr is not None and impulse_size >= self.config.min_impulse_atr * atr:
                self.state = MarketState.COUNTERMOVE_ARMED
                episode = PullbackEpisode(
                    episode_id=f"{bar.ts.isoformat()}-{self.direction_label()}",
                    engine_version=ENGINE_VERSION,
                    side_sign=self.side_sign,
                    impulse_extreme=float(self._impulse_extreme or bar.close),
                    armed_ts=bar.ts,
                )
                episode.record(MarketState.COUNTERMOVE_ARMED, bar.ts, bar.close, 0.0)
                self.episodes.append(episode)
                self._countermove_extreme = bar.low if self.side_sign > 0 else bar.high

    def _run_armed(self, bar: M5Bar) -> None:
        self._track_countermove_extreme(bar)
        atr = self._m5_atr()
        if atr is None:
            return
        depth = self._countermove_depth()
        depth_pct = depth / bar.close * 100.0 if bar.close else 0.0
        atr_pct = atr / bar.close * 100.0 if bar.close else 0.0
        threshold_pct = max(
            self.config.countermove_min_pct,
            self.config.countermove_atr_fraction * atr_pct,
        )
        if depth_pct >= threshold_pct:
            self._transition_episode(MarketState.COUNTERMOVE_ACTIVE, bar, depth / atr)
            return
        # A single adverse candle is an arm, not proof: a fresh aligned
        # extreme cancels the armed state (sec 16.4).
        if self._impulse_extreme is not None:
            edge = bar.high if self.side_sign > 0 else bar.low
            if self.side_sign * (edge - self._impulse_extreme) > 0:
                self._abort_episode(bar)

    def _run_active(self, bar: M5Bar) -> None:
        prev_extreme = self._countermove_extreme
        self._track_countermove_extreme(bar)
        atr = self._m5_atr()
        if atr is None:
            return
        depth = self._countermove_depth()
        episode = self.active_episode
        if self._check_failure(bar, atr, depth):
            return
        if episode is None:
            self.state = MarketState.RANGE
            return
        bars_in_episode = self._bars_since(episode.armed_ts)
        if bars_in_episode < self.config.min_episode_bars:
            return
        # Stabilizing: no strictly deeper adverse extreme this bar plus an
        # aligned close (higher low + up close for a bull pullback).
        adverse_edge = bar.low if self.side_sign > 0 else bar.high
        made_new_adverse = prev_extreme is None or self.side_sign * (prev_extreme - adverse_edge) > 0
        prev_close = self.bars[-2].close if len(self.bars) >= 2 else bar.close
        aligned_close = self.side_sign * (bar.close - prev_close) > 0
        if not made_new_adverse and aligned_close:
            self._stabilize_pivot = bar.high if self.side_sign > 0 else bar.low
            self._transition_episode(MarketState.STABILIZING, bar, depth / atr)

    def _run_stabilizing(self, bar: M5Bar) -> None:
        atr = self._m5_atr()
        if atr is None:
            return
        depth = self._countermove_depth()
        if self._check_failure(bar, atr, depth):
            return
        # A deeper adverse extreme sends the episode back to ACTIVE.
        adverse_edge = bar.low if self.side_sign > 0 else bar.high
        if (
            self._countermove_extreme is not None
            and self.side_sign * (self._countermove_extreme - adverse_edge) > 0
        ):
            self._track_countermove_extreme(bar)
            self._transition_episode(MarketState.COUNTERMOVE_ACTIVE, bar, self._countermove_depth() / atr)
            return

        ema8 = self._ema(8)
        recovered = 0.0
        if self._impulse_extreme is not None and self._countermove_extreme is not None:
            total = self.side_sign * (self._impulse_extreme - self._countermove_extreme)
            if total > 0:
                recovered = self.side_sign * (bar.close - self._countermove_extreme) / total
        pivot_break = (
            self._stabilize_pivot is not None
            and self.side_sign * (bar.close - self._stabilize_pivot) > 0
        )
        ema_reclaim = ema8 is None or self.side_sign * (bar.close - ema8) >= 0
        if pivot_break and ema_reclaim and recovered >= self.config.resume_recovery_fraction:
            episode = self.active_episode
            if episode is not None:
                episode.record(MarketState.TREND_RESUMED, bar.ts, bar.close, depth / atr)
                episode.outcome = "RESUMED"
            self.state = MarketState.TREND_RESUMED
            self._resume_cooldown = 3
            self._countermove_extreme = None
            self._stabilize_pivot = None

    def _check_failure(self, bar: M5Bar, atr: float, depth: float) -> bool:
        episode = self.active_episode
        fail_reasons = []
        if depth > self.config.fail_depth_atr * atr:
            fail_reasons.append("depth")
        if self._vwap is not None:
            adverse_closes = 0
            for candidate in reversed(self.bars[-self.config.fail_vwap_closes:]):
                if self.side_sign * (candidate.close - self._vwap) < -self.config.fail_vwap_atr * atr:
                    adverse_closes += 1
            if adverse_closes >= self.config.fail_vwap_closes:
                fail_reasons.append("vwap")
        if episode is not None and self._bars_since(episode.armed_ts) > self.config.fail_max_bars:
            fail_reasons.append("duration")
        if not fail_reasons:
            return False
        if episode is not None:
            episode.record(MarketState.REGIME_FAILED, bar.ts, bar.close, depth / atr if atr else 0.0)
            episode.outcome = "FAILED"
        self.state = MarketState.REGIME_FAILED
        self.side_sign = 0
        self._impulse_extreme = None
        self._impulse_start_price = None
        self._countermove_extreme = None
        self._stabilize_pivot = None
        return True

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def direction_label(self) -> str:
        if self.side_sign > 0:
            return "BULL"
        if self.side_sign < 0:
            return "BEAR"
        return "NONE"

    def _ema(self, length: int) -> float | None:
        closes = [b.close for b in self.bars]
        if len(closes) < length:
            return None
        k = 2.0 / (length + 1)
        ema = sum(closes[:length]) / length
        for close in closes[length:]:
            ema = close * k + ema * (1 - k)
        return ema

    def _track_countermove_extreme(self, bar: M5Bar) -> None:
        adverse_edge = bar.low if self.side_sign > 0 else bar.high
        if (
            self._countermove_extreme is None
            or self.side_sign * (self._countermove_extreme - adverse_edge) > 0
        ):
            self._countermove_extreme = adverse_edge

    def _bars_since(self, ts: datetime) -> int:
        return sum(1 for b in self.bars if b.ts > ts)

    def _transition_episode(self, state: MarketState, bar: M5Bar, depth_atr: float) -> None:
        self.state = state
        episode = self.active_episode
        if episode is not None:
            episode.record(state, bar.ts, bar.close, depth_atr)

    def _abort_episode(self, bar: M5Bar) -> None:
        episode = self.active_episode
        if episode is not None:
            episode.record(
                MarketState.BULL_IMPULSE if self.side_sign > 0 else MarketState.BEAR_IMPULSE,
                bar.ts,
                bar.close,
                0.0,
            )
            episode.outcome = "ABORTED"
        self.state = MarketState.BULL_IMPULSE if self.side_sign > 0 else MarketState.BEAR_IMPULSE
        self._countermove_extreme = None

    def _make_snapshot(self, ts: datetime | None, stale: bool) -> MarketStateSnapshot:
        atr = self._m5_atr()
        depth_atr = 0.0
        if atr and self.state in (
            MarketState.COUNTERMOVE_ARMED,
            MarketState.COUNTERMOVE_ACTIVE,
            MarketState.STABILIZING,
        ):
            depth_atr = self._countermove_depth() / atr
        close = self.bars[-1].close if self.bars else self.prior_close
        score_side = self.side_sign if self.side_sign else (1 if self._day_return_pct(close) >= 0 else -1)
        return MarketStateSnapshot(
            state=self.state,
            ts=ts,
            side_sign=self.side_sign,
            trend_score=self._trend_score(score_side, self.bars[-1]) if self.bars else 0.0,
            day_return_pct=self._day_return_pct(close),
            vwap=self._vwap,
            m5_atr=atr,
            countermove_depth_atr=depth_atr,
            stale=stale,
        )


def run_session(
    bars: list[M5Bar],
    prior_close: float,
    config: MarketStateConfig | None = None,
    daily_atr_pct: float | None = None,
) -> tuple[list[MarketStateSnapshot], MarketStateEngine]:
    """Feed a full session; returns per-bar snapshots plus the engine."""
    engine = MarketStateEngine(prior_close, config=config, daily_atr_pct=daily_atr_pct)
    snapshots = [engine.on_bar(bar) for bar in bars]
    return snapshots, engine


def mirror_bar(bar: M5Bar, pivot: float) -> M5Bar:
    """Reflect a bar around `pivot` (price axis); highs and lows swap roles."""
    return M5Bar(
        ts=bar.ts,
        open=2 * pivot - bar.open,
        high=2 * pivot - bar.low,
        low=2 * pivot - bar.high,
        close=2 * pivot - bar.close,
        volume=bar.volume,
        complete=bar.complete,
    )


_MIRROR_STATE = {
    MarketState.BULL_IMPULSE: MarketState.BEAR_IMPULSE,
    MarketState.BEAR_IMPULSE: MarketState.BULL_IMPULSE,
}


def mirror_state(state: MarketState) -> MarketState:
    return _MIRROR_STATE.get(state, state)
