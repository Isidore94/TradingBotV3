"""Shadow bridge (plan.md sec 16 champion/challenger): bot bars -> engine."""

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import market_state_bridge as bridge  # noqa: E402
from market_state import MarketState  # noqa: E402


@dataclass
class BotBar:
    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 1_000_000.0


SESSION_START = datetime(2026, 7, 10, 9, 30)
PRIOR_CLOSE = 500.0


def bot_bars(closes, rng=0.5):
    bars = []
    prev = PRIOR_CLOSE
    for i, close in enumerate(closes):
        bars.append(
            BotBar(
                dt=SESSION_START + timedelta(minutes=5 * i),
                open=prev,
                high=max(prev, close) + rng,
                low=min(prev, close) - rng,
                close=close,
            )
        )
        prev = close
    return bars


RALLY = [500.2, 500.4, 500.5, 501.0, 502.0, 502.8, 503.5, 504.2, 505.0, 505.8, 506.5, 507.0]


def test_conversion_marks_only_a_forming_last_bar_incomplete():
    bars = bot_bars(RALLY)
    now_mid_bar = bars[-1].dt + timedelta(minutes=2)
    converted = bridge.m5_bars_from_bot_bars(bars, now=now_mid_bar)
    assert all(b.complete for b in converted[:-1])
    assert converted[-1].complete is False, "a forming bar must not drive state"

    now_after_close = bars[-1].dt + timedelta(minutes=6)
    converted = bridge.m5_bars_from_bot_bars(bars, now=now_after_close)
    assert converted[-1].complete is True
    assert converted[-1].ts.replace(tzinfo=None) == bars[-1].dt + timedelta(minutes=5)
    assert converted[-1].ts.utcoffset() is not None


def test_shadow_state_reaches_impulse_on_a_trend_day():
    now = SESSION_START + timedelta(minutes=5 * len(RALLY) + 6)
    snapshot = bridge.evaluate_spy_shadow_state(bot_bars(RALLY), PRIOR_CLOSE, now=now)
    assert snapshot is not None
    assert snapshot.state in (MarketState.BULL_IMPULSE, MarketState.COUNTERMOVE_ARMED)
    assert snapshot.side_sign == 1


def test_record_appends_only_on_change_and_flags_agreement(tmp_path, monkeypatch):
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(bridge, "shadow_log_path", lambda: log)
    bridge.reset_shadow_dedupe()

    bars = bot_bars(RALLY)
    now = bars[-1].dt + timedelta(minutes=6)

    first = bridge.record_spy_shadow(bars, PRIOR_CLOSE, legacy_pause_start=None, side="long", now=now)
    second = bridge.record_spy_shadow(bars, PRIOR_CLOSE, legacy_pause_start=None, side="long", now=now)
    assert first is not None and second is not None

    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1, "unchanged state must not spam the shadow log"
    assert rows[0]["schema"] == "spy_state_shadow_v2"
    assert rows[0]["config_hash"] and rows[0]["machine"]
    assert rows[0]["evaluated_at"] and rows[0]["bar_ts"]
    assert datetime.fromisoformat(rows[0]["evaluated_at"]).utcoffset() is not None
    assert datetime.fromisoformat(rows[0]["bar_ts"]).utcoffset() is not None
    assert rows[0]["timezone"]
    assert rows[0]["observation_id"].startswith("SPY|2026-07-10|")
    assert rows[0]["engine_paused"] is False
    assert rows[0]["legacy_paused"] is False
    assert rows[0]["agree"] is True

    # legacy suddenly claims a pause (single red candle rule) -> divergence row
    bridge.record_spy_shadow(bars, PRIOR_CLOSE, legacy_pause_start=now, side="long", now=now)
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[1]["legacy_paused"] is True and rows[1]["agree"] is False
    status = json.loads(bridge.shadow_status_path().read_text(encoding="utf-8"))
    assert status["schema"] == "spy_state_shadow_status_v1"
    assert status["evaluations"] == 3
    assert status["usable_evaluations"] == 3
    assert status["rows_written"] == 2
    assert status["errors"] == 0


def test_record_never_raises_on_garbage(monkeypatch):
    assert bridge.record_spy_shadow(None, None) is None
    assert bridge.record_spy_shadow([], 0.0) is None

    class ExplodingBar:
        @property
        def dt(self):
            raise RuntimeError("boom")

    assert bridge.record_spy_shadow([ExplodingBar()], 500.0) is None


def test_legacy_spy_log_is_archived_before_v2_append(tmp_path, monkeypatch):
    log = tmp_path / "spy_state_shadow.jsonl"
    log.write_text('{"ts":"2026-07-10T09:30:00","state":"RANGE"}\n', encoding="utf-8")
    monkeypatch.setattr(bridge, "shadow_log_path", lambda: log)
    bridge.reset_shadow_dedupe()
    bars = bot_bars(RALLY)
    now = bars[-1].dt + timedelta(minutes=6)

    bridge.record_spy_shadow(bars, PRIOR_CLOSE, legacy_pause_start=None, side="long", now=now)

    archives = list(tmp_path.glob("spy_state_shadow.legacy-*.jsonl"))
    assert len(archives) == 1
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["schema"] == "spy_state_shadow_v2"
