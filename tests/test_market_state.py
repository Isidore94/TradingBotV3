"""Packet B (plan.md sec 16/24/25): pure SPY market-state machine."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from market_state import (  # noqa: E402
    M5Bar,
    MarketState,
    MarketStateEngine,
    mirror_bar,
    mirror_state,
    run_session,
)

PRIOR_CLOSE = 500.0
SESSION_START = datetime(2026, 7, 10, 9, 35)


def make_bars(closes, rng=0.5, start=SESSION_START, volume=1_000_000):
    bars = []
    prev_close = PRIOR_CLOSE
    for i, close in enumerate(closes):
        open_ = prev_close
        bars.append(
            M5Bar(
                ts=start + timedelta(minutes=5 * i),
                open=open_,
                high=max(open_, close) + rng,
                low=min(open_, close) - rng,
                close=close,
                volume=volume,
            )
        )
        prev_close = close
    return bars


# Opening drift, sustained impulse, orderly two-bar pullback, higher-low
# stabilization, then a pivot-break resumption.
BULL_LIFECYCLE_CLOSES = [
    500.2, 500.4, 500.5,                      # opening discovery
    501.0, 502.0, 502.8, 503.5, 504.2,        # impulse builds
    505.0, 505.8, 506.5, 507.0,               # extends (extreme 507.5 high)
    506.6, 506.1, 505.9,                      # counter-move (armed -> active)
    506.3,                                    # higher low + up close -> stabilizing
    507.2,                                    # pivot break -> trend resumed
]


def states_for(closes, **kwargs):
    snapshots, engine = run_session(make_bars(closes, **kwargs), PRIOR_CLOSE)
    return [s.state for s in snapshots], engine


def test_bull_lifecycle_impulse_pullback_stabilize_resume():
    states, engine = states_for(BULL_LIFECYCLE_CLOSES)

    assert MarketState.BULL_IMPULSE in states
    assert MarketState.COUNTERMOVE_ARMED in states
    assert MarketState.COUNTERMOVE_ACTIVE in states
    assert MarketState.STABILIZING in states
    assert states[-1] == MarketState.TREND_RESUMED

    # lifecycle ordering
    order = [
        states.index(MarketState.BULL_IMPULSE),
        states.index(MarketState.COUNTERMOVE_ARMED),
        states.index(MarketState.COUNTERMOVE_ACTIVE),
        states.index(MarketState.STABILIZING),
        states.index(MarketState.TREND_RESUMED),
    ]
    assert order == sorted(order)

    assert len(engine.episodes) == 1
    episode = engine.episodes[0]
    assert episode.outcome == "RESUMED"
    assert episode.direction == "BULL_PULLBACK"
    assert [e.state for e in episode.events][0] == MarketState.COUNTERMOVE_ARMED
    assert episode.events[-1].state == MarketState.TREND_RESUMED


def test_bull_failure_when_pullback_runs_too_deep():
    closes = BULL_LIFECYCLE_CLOSES[:15] + [505.0, 504.2, 503.4, 502.6]
    states, engine = states_for(closes)
    assert MarketState.COUNTERMOVE_ACTIVE in states
    assert MarketState.REGIME_FAILED in states
    assert engine.episodes and engine.episodes[-1].outcome == "FAILED"


def test_bearish_mirror_of_every_sequence():
    for closes in (BULL_LIFECYCLE_CLOSES, BULL_LIFECYCLE_CLOSES[:15] + [505.0, 504.2, 503.4, 502.6]):
        bull_bars = make_bars(closes)
        bear_bars = [mirror_bar(b, PRIOR_CLOSE) for b in bull_bars]
        bull_states = [s.state for s in run_session(bull_bars, PRIOR_CLOSE)[0]]
        bear_states = [s.state for s in run_session(bear_bars, PRIOR_CLOSE)[0]]
        assert bear_states == [mirror_state(s) for s in bull_states]


def test_chop_never_creates_an_episode():
    closes = [500.0 + (0.3 if i % 2 else -0.3) for i in range(30)]
    states, engine = states_for(closes)
    assert MarketState.COUNTERMOVE_ACTIVE not in states
    assert not engine.episodes
    assert states[-1] in (MarketState.RANGE, MarketState.OPENING_DISCOVERY)


def test_single_red_candle_does_not_prove_a_pullback():
    # One shallow red bar inside a strong impulse, then fresh highs: at most
    # an armed state that aborts; never an ACTIVE counter-move.
    closes = BULL_LIFECYCLE_CLOSES[:12] + [506.6, 507.4, 508.0, 508.6]
    states, engine = states_for(closes)
    assert MarketState.COUNTERMOVE_ACTIVE not in states
    assert states[-1] == MarketState.BULL_IMPULSE
    assert all(e.outcome in ("", "ABORTED") for e in engine.episodes)


def test_incomplete_bar_cannot_transition_state():
    bars = make_bars(BULL_LIFECYCLE_CLOSES[:12])
    engine = MarketStateEngine(PRIOR_CLOSE)
    for bar in bars:
        engine.on_bar(bar)
    state_before = engine.state
    partial = M5Bar(
        ts=bars[-1].ts + timedelta(minutes=5),
        open=507.0,
        high=507.2,
        low=490.0,
        close=490.5,
        volume=1_000_000,
        complete=False,
    )
    snapshot = engine.on_bar(partial)
    assert snapshot.stale is True
    assert engine.state == state_before


def test_gapped_stale_bar_cannot_transition_state():
    bars = make_bars(BULL_LIFECYCLE_CLOSES[:12])
    engine = MarketStateEngine(PRIOR_CLOSE)
    for bar in bars:
        engine.on_bar(bar)
    state_before = engine.state
    gapped = M5Bar(
        ts=bars[-1].ts + timedelta(minutes=45),
        open=507.0,
        high=507.2,
        low=490.0,
        close=490.5,
        volume=1_000_000,
    )
    snapshot = engine.on_bar(gapped)
    assert snapshot.stale is True
    assert engine.state == state_before


def test_snapshot_reports_depth_and_direction_during_episode():
    bars = make_bars(BULL_LIFECYCLE_CLOSES[:15])
    engine = MarketStateEngine(PRIOR_CLOSE)
    last = None
    for bar in bars:
        last = engine.on_bar(bar)
    assert last is not None
    assert last.state == MarketState.COUNTERMOVE_ACTIVE
    assert last.side_sign == 1
    assert last.countermove_depth_atr > 0
    assert last.vwap is not None
