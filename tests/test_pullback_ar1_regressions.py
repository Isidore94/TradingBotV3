"""Extra AR-1 regressions that drive the two previously hidden branches."""

from __future__ import annotations

import dataclasses
from datetime import timedelta

from test_pct1_pullback_alert import (
    OPEN_GATE_D1_ATR,
    M15_LONG_CLOSES,
    M15_LATE_FLAG_CLOSES,
    M15_LATE_FLAG_CROSS_INDEX,
    M15_LATE_FLAG_RECLAIM_INDEX,
    M15_RECLAIM_INDEX,
    M30_HOLD_CLOSES,
    M30_HOLD_CROSS_INDEX,
    M30_HOLD_RECLAIM_INDEX,
    M30_SMA,
    TRIGGER_H1,
    TRIGGER_RECLAIM,
    TRIGGER_THEN_LRSI,
    bar_dt,
    bar_end,
    make_bars,
    one_fire,
)
from test_pct1_pullback_desk import (
    WATCH_KIND,
    _events,
    _install_stub_caches,
    _panel,
    settle_pullback,
)
from test_ws_10c_h1_retester import (
    GOLDEN_CONFIRM_DT,
    golden_long_h1_bars,
    golden_m5_series,
)


def test_prearm_native_cross_does_not_block_a_later_eligible_m15_companion():
    """The native branch is present, but its pre-arm cross is not eligible."""
    from indicators.pullback_sma_reclaim import evaluate

    primary = make_bars(M30_HOLD_CLOSES[: M30_HOLD_CROSS_INDEX + 4], 30)
    companion = make_bars(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)
    armed_at = bar_end(M30_HOLD_CROSS_INDEX, 30) + timedelta(minutes=1)
    result = evaluate(
        primary,
        side="long",
        sma_length=M30_SMA,
        bar_minutes=30,
        armed_at=armed_at,
        now=bar_end(M30_HOLD_CROSS_INDEX + 3, 30),
        companion_bars=companion,
        companion_minutes=15,
        d1_atr=OPEN_GATE_D1_ATR,
    )

    assert bar_end(M30_HOLD_RECLAIM_INDEX, 30) < armed_at
    fire = one_fire(result, TRIGGER_THEN_LRSI)
    assert fire.cross_timeframe == "M15"
    assert fire.cross_bar_dt == bar_dt(M15_RECLAIM_INDEX, 15)
    assert fire.bar_dt == bar_dt(M15_RECLAIM_INDEX, 15)


def test_later_companion_cross_after_a_prearm_one_is_still_eligible(monkeypatch):
    """Companion selection itself applies the arm fence, not only its caller."""
    from indicators import pullback_sma_reclaim as rule

    class _Lrsi:
        def __init__(self, count, crosses):
            self.values = [90.0] * count
            self._crosses = crosses

        def cross_up_indices(self, _level):
            return self._crosses

        def cross_down_indices(self, _level):
            # v2 dip gate: one M30 bear flip on the reclaim bar, none on M15.
            return (M30_HOLD_RECLAIM_INDEX,) if len(self.values) <= 150 else ()

    primary = make_bars(M30_HOLD_CLOSES[: M30_HOLD_CROSS_INDEX + 4], 30)
    companion = make_bars(M15_LONG_CLOSES[:215], 15)
    monkeypatch.setattr(
        rule, "_lrsi_series", lambda closes, _side: _Lrsi(len(closes), (210, 214) if len(closes) > 150 else ())
    )
    armed_at = bar_end(210, 15) + timedelta(minutes=1)
    result = rule.evaluate(
        primary,
        side="long",
        sma_length=M30_SMA,
        bar_minutes=30,
        armed_at=armed_at,
        now=bar_end(M30_HOLD_CROSS_INDEX + 3, 30),
        companion_bars=companion,
        companion_minutes=15,
        d1_atr=OPEN_GATE_D1_ATR,
    )

    fire = one_fire(result, TRIGGER_THEN_LRSI)
    assert fire.cross_bar_dt == bar_dt(214, 15)


def test_m30_hold_selects_the_cross_that_finishes_first(monkeypatch):
    """Candidate order is completion time, then a deterministic start/label tie."""
    from indicators import pullback_sma_reclaim as rule

    class _Lrsi:
        def __init__(self, count, crosses):
            self.values = [90.0] * count
            self._crosses = crosses

        def cross_up_indices(self, _level):
            return self._crosses

        def cross_down_indices(self, _level):
            # v2 dip gate: one M30 bear flip on the reclaim bar, none on M15.
            return (M30_HOLD_RECLAIM_INDEX,) if len(self.values) <= 150 else ()

    primary = make_bars(M30_HOLD_CLOSES[: M30_HOLD_CROSS_INDEX + 4], 30)
    companion = make_bars(M15_LONG_CLOSES[:215], 15)
    monkeypatch.setattr(
        rule, "_lrsi_series", lambda closes, _side: _Lrsi(len(closes), (211,) if len(closes) > 150 else (105,))
    )
    # A ten-minute companion starts later but closes before the native M30.
    result = rule.evaluate(
        primary,
        side="long",
        sma_length=M30_SMA,
        bar_minutes=30,
        armed_at=bar_dt(0, 30),
        now=bar_end(M30_HOLD_CROSS_INDEX + 3, 30),
        companion_bars=companion,
        companion_minutes=10,
        d1_atr=OPEN_GATE_D1_ATR,
    )

    fire = one_fire(result, TRIGGER_THEN_LRSI)
    assert fire.cross_timeframe == "M10"
    assert fire.bar_dt == bar_dt(211, 15)


def test_persisted_aware_stamp_compares_against_naive_market_time_by_instant():
    from ui.panels.alert_center_panel import AlertCenterPanel

    assert AlertCenterPanel._pullback_mark_covers(
        "2026-08-27T15:00:00+00:00", "2026-08-27T08:00:00"
    )
    assert not AlertCenterPanel._pullback_mark_covers(
        "2026-08-27T14:59:00+00:00", "2026-08-27T08:00:00"
    )


def test_real_worker_evidence_keeps_a_prior_lrsi_cross_separate_from_event_bar(
    monkeypatch, tmp_path
):
    """A reclaim event and its allowed prior cross retain both timestamps."""
    m15 = make_bars(M15_LATE_FLAG_CLOSES[: M15_LATE_FLAG_RECLAIM_INDEX + 1], 15)
    _install_stub_caches(monkeypatch, bars={("NVDA", 15): m15})
    panel = _panel(monkeypatch, tmp_path)
    panel.arm_chart_watch_for("NVDA", "LONG", WATCH_KIND)
    panel._chart_watches = [
        dataclasses.replace(
            panel._chart_watches[0],
            armed_at=bar_dt(0, 15),
            triggers=(TRIGGER_RECLAIM,),
        )
    ]

    panel._poll_pullback_watches(now=bar_end(M15_LATE_FLAG_RECLAIM_INDEX, 15))
    settle_pullback(panel)

    detail = _events(tmp_path, "watch_fired")[0]["detail"]
    assert detail["bar_dt"] == bar_dt(M15_LATE_FLAG_RECLAIM_INDEX, 15).isoformat()
    assert detail["cross_bar_dt"] == bar_dt(M15_LATE_FLAG_CROSS_INDEX, 15).isoformat()


def test_actual_h1_poll_preserves_the_trigger_and_timeframe_in_both_outputs(
    monkeypatch, tmp_path
):
    panel = _panel(monkeypatch, tmp_path)
    h1_bars, _ = golden_long_h1_bars()
    monkeypatch.setattr(
        panel, "_m5_bars_for", lambda _symbol, **_kwargs: golden_m5_series(h1_bars)
    )
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    watch = panel._chart_watches[0]
    panel._chart_watches = [
        dataclasses.replace(
            watch,
            armed_at=GOLDEN_CONFIRM_DT - timedelta(days=1),
            triggers=(TRIGGER_H1,),
        )
    ]

    panel._poll_pullback_watches(now=GOLDEN_CONFIRM_DT + timedelta(hours=1))

    assert panel._alerts[0].payload["trigger"] == TRIGGER_H1
    assert panel._alerts[0].payload["timeframe"] == "H1"
    row = _events(tmp_path, "watch_fired")[0]["detail"]
    assert row["trigger"] == TRIGGER_H1
    assert row["timeframe"] == "H1"
