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


def read_status():
    return json.loads(bridge.shadow_status_path().read_text(encoding="utf-8"))


def read_log(tmp_path, name="shadow.jsonl"):
    log = tmp_path / name
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def gapped_bars(closes, gap_minutes=30):
    """Same session, but the final bar arrives after a feed gap."""
    bars = bot_bars(closes)
    last = bars[-1]
    bars[-1] = BotBar(
        dt=last.dt + timedelta(minutes=gap_minutes),
        open=last.open,
        high=last.high,
        low=last.low,
        close=last.close,
    )
    return bars


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
    assert rows[0]["schema"] == bridge.SHADOW_SCHEMA
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
    status = read_status()
    assert status["schema"] == bridge.STATUS_SCHEMA
    assert status["evaluations"] == 3
    assert status["usable_evaluations"] == 3
    assert status["rows_written"] == 2
    assert status["errors"] == 0
    assert status["stale_evaluations"] == 0
    assert status["incomplete_bar_evaluations"] == 0
    assert status["gap_stale_evaluations"] == 0


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
    assert len(rows) == 1 and rows[0]["schema"] == bridge.SHADOW_SCHEMA


def test_v2_rows_are_kept_in_place_by_the_additive_v3_bump(tmp_path, monkeypatch):
    """v3 only adds fields, so a live v2 log must not be rotated away."""
    log = tmp_path / "spy_state_shadow.jsonl"
    log.write_text(
        json.dumps({"schema": "spy_state_shadow_v2", "state": "RANGE"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(bridge, "shadow_log_path", lambda: log)
    bridge.reset_shadow_dedupe()
    bars = bot_bars(RALLY)

    bridge.record_spy_shadow(bars, PRIOR_CLOSE, now=bars[-1].dt + timedelta(minutes=6))

    assert not list(tmp_path.glob("spy_state_shadow.legacy-*.jsonl"))
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [row["schema"] for row in rows] == ["spy_state_shadow_v2", bridge.SHADOW_SCHEMA]


# ---------------------------------------------------------------------------
# (a) a forming bar is not a completed bar (plan.md sec 5)
# ---------------------------------------------------------------------------
def test_forming_bar_never_advances_completed_bar_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "shadow_log_path", lambda: tmp_path / "shadow.jsonl")
    bridge.reset_shadow_dedupe()

    settled = bot_bars(RALLY)
    complete_now = settled[-1].dt + timedelta(minutes=6)
    bridge.record_spy_shadow(settled, PRIOR_CLOSE, now=complete_now)

    after_complete = read_status()
    last_complete = after_complete["last_complete_bar_at"]
    assert after_complete["usable_evaluations"] == 1
    assert last_complete == read_log(tmp_path)[0]["bar_ts"]

    # A newer bar that is still forming: the engine refuses it, so the audit
    # must not claim a newer completed bar or another usable evaluation.
    forming = settled + [
        BotBar(dt=settled[-1].dt + timedelta(minutes=5), open=507.0, high=507.6, low=506.9, close=507.4)
    ]
    bridge.record_spy_shadow(forming, PRIOR_CLOSE, now=forming[-1].dt + timedelta(minutes=2))

    status = read_status()
    assert status["evaluations"] == 2
    assert status["usable_evaluations"] == 1, "a forming bar is not a usable evaluation"
    assert status["last_complete_bar_at"] == last_complete, "forming bar advanced the completed stamp"
    assert status["stale_evaluations"] == 1
    assert status["incomplete_bar_evaluations"] == 1
    assert status["gap_stale_evaluations"] == 0
    assert status["last_stale_reason"] == "incomplete_bar"
    # The snapshot really did move to the forming bar - that is why the naive
    # "snapshot is not None" accounting was wrong.
    assert status["last_snapshot_bar_at"] > status["last_complete_bar_at"]


# ---------------------------------------------------------------------------
# (b) incomplete-bar staleness and gap staleness are separable
# ---------------------------------------------------------------------------
def test_incomplete_and_gap_staleness_are_counted_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "shadow_log_path", lambda: tmp_path / "shadow.jsonl")
    bridge.reset_shadow_dedupe()

    gapped = gapped_bars(RALLY)
    # Complete final bar, but it lands 30 minutes after the previous one.
    gap_only = bridge.evaluate_spy_shadow(gapped, PRIOR_CLOSE, now=gapped[-1].dt + timedelta(minutes=6))
    assert gap_only.snapshot.stale is True
    assert gap_only.gap_stale is True and gap_only.incomplete_bar is False
    assert gap_only.stale_reason == "bar_gap"
    assert gap_only.is_usable is False
    # The completed-bar stamp falls back to the last bar the engine consumed.
    assert gap_only.last_complete_bar_ts.replace(tzinfo=None) == (
        gapped[-2].dt + timedelta(minutes=5)
    )

    forming_only = bridge.evaluate_spy_shadow(
        bot_bars(RALLY), PRIOR_CLOSE, now=SESSION_START + timedelta(minutes=5 * (len(RALLY) - 1) + 2)
    )
    assert forming_only.incomplete_bar is True and forming_only.gap_stale is False
    assert forming_only.stale_reason == "incomplete_bar"

    both = bridge.evaluate_spy_shadow(gapped, PRIOR_CLOSE, now=gapped[-1].dt + timedelta(minutes=2))
    assert both.incomplete_bar is True and both.gap_stale is True
    assert both.stale_reason == "incomplete_bar+bar_gap"

    fresh = bridge.evaluate_spy_shadow(
        bot_bars(RALLY), PRIOR_CLOSE, now=SESSION_START + timedelta(minutes=5 * len(RALLY) + 1)
    )
    assert fresh.is_usable is True and fresh.stale_reason == ""
    assert fresh.last_complete_bar_ts == fresh.snapshot.ts

    # ... and the counters keep them apart on disk.
    bridge.record_spy_shadow(gapped, PRIOR_CLOSE, now=gapped[-1].dt + timedelta(minutes=6))
    bridge.record_spy_shadow(gapped, PRIOR_CLOSE, now=gapped[-1].dt + timedelta(minutes=2))
    status = read_status()
    assert status["gap_stale_evaluations"] == 2
    assert status["incomplete_bar_evaluations"] == 1
    assert status["stale_evaluations"] == 2
    assert status["usable_evaluations"] == 0

    rows = read_log(tmp_path)
    assert rows[0]["stale"] is True
    assert rows[0]["stale_reason"] == "bar_gap"
    assert rows[0]["gap_stale"] is True and rows[0]["incomplete_bar"] is False
    assert rows[0]["usable"] is False
    assert rows[0]["complete_bar_ts"] and rows[0]["complete_bar_ts"] < rows[0]["bar_ts"]


def test_evaluation_flags_agree_with_the_engine_gate():
    """Observability only: the split flags must reproduce engine.stale exactly."""
    now_variants = [
        SESSION_START + timedelta(minutes=5 * len(RALLY) + 1),
        SESSION_START + timedelta(minutes=5 * (len(RALLY) - 1) + 2),
    ]
    for source in (bot_bars(RALLY), gapped_bars(RALLY)):
        for moment in now_variants + [source[-1].dt + timedelta(minutes=2), source[-1].dt + timedelta(minutes=6)]:
            evaluation = bridge.evaluate_spy_shadow(source, PRIOR_CLOSE, now=moment)
            merged = evaluation.incomplete_bar or evaluation.gap_stale
            assert evaluation.snapshot.stale is merged
            # Back-compat: the snapshot-only helper is unchanged.
            assert bridge.evaluate_spy_shadow_state(source, PRIOR_CLOSE, now=moment) == evaluation.snapshot


# ---------------------------------------------------------------------------
# (c) dedupe is scoped to the session
# ---------------------------------------------------------------------------
def test_new_session_first_row_is_never_deduped(tmp_path, monkeypatch):
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(bridge, "shadow_log_path", lambda: log)
    bridge.reset_shadow_dedupe()

    day_one = bot_bars(RALLY)
    now_one = day_one[-1].dt + timedelta(minutes=6)
    bridge.record_spy_shadow(day_one, PRIOR_CLOSE, now=now_one)
    bridge.record_spy_shadow(day_one, PRIOR_CLOSE, now=now_one)  # same session: deduped
    assert len(read_log(tmp_path)) == 1

    # Next session, identical shape -> identical fingerprint. A process that
    # stayed alive across midnight must still record the new session.
    day_two = [
        BotBar(
            dt=bar.dt + timedelta(days=1),
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
        )
        for bar in day_one
    ]
    now_two = day_two[-1].dt + timedelta(minutes=6)
    bridge.record_spy_shadow(day_two, PRIOR_CLOSE, now=now_two)

    rows = read_log(tmp_path)
    assert len(rows) == 2, "a new session's first observation was deduped away"
    assert [row["session_date"] for row in rows] == ["2026-07-10", "2026-07-11"]
    assert rows[0]["state"] == rows[1]["state"], "the fingerprints really were identical"

    # ... and the second session dedupes normally from there.
    bridge.record_spy_shadow(day_two, PRIOR_CLOSE, now=now_two)
    assert len(read_log(tmp_path)) == 2

    status = read_status()
    assert status["session_date"] == "2026-07-11"
    assert status["rows_written"] == 1, "coverage rolls over with the session"


def test_a_failed_append_does_not_suppress_the_retry(tmp_path, monkeypatch):
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setattr(bridge, "shadow_log_path", lambda: log)
    bridge.reset_shadow_dedupe()
    bars = bot_bars(RALLY)
    now = bars[-1].dt + timedelta(minutes=6)

    real_append = bridge.append_jsonl_rows
    monkeypatch.setattr(
        bridge,
        "append_jsonl_rows",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    assert bridge.record_spy_shadow(bars, PRIOR_CLOSE, now=now) is None
    assert read_status()["errors"] == 1

    monkeypatch.setattr(bridge, "append_jsonl_rows", real_append)
    assert bridge.record_spy_shadow(bars, PRIOR_CLOSE, now=now) is not None
    assert len(read_log(tmp_path)) == 1
