"""AR-1 delivery contracts for asynchronous Pullback cache refreshes."""

from __future__ import annotations

import dataclasses
import sys
from datetime import timedelta
from pathlib import Path

from test_pct1_pullback_alert import (  # noqa: E402
    M15_LONG_CLOSES,
    M15_RECLAIM_INDEX,
    TRIGGER_H1,
    TRIGGER_RECLAIM,
    bar_end,
    make_bars,
)
from test_pct1_pullback_desk import (  # noqa: E402
    WATCH_KIND,
    _events,
    _panel,
    _qt_app,
    settle_pullback,
)
from test_ws_10c_h1_retester import (  # noqa: E402
    GOLDEN_CONFIRM_DT,
    golden_long_h1_bars,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


class _DelayedCache:
    """The cache's real read/request surface with delivery under test control."""

    def __init__(self, interval_minutes=60, **_kwargs):
        self.interval_minutes = int(interval_minutes)
        self.requests: list[object] = []
        self.rows: dict[str, list[dict]] = {}

    def bars_for(self, symbol):
        return list(self.rows.get(str(symbol).upper(), ()))

    def request(self, symbol, *, now=None):
        self.requests.append((str(symbol).upper(), now))
        return len(self.requests) == 1

    def unavailable(self, _symbol):
        return False

    def last_refresh_failed(self, _symbol):
        return False


def _manual_pullback_panel(monkeypatch, tmp_path):
    _qt_app()
    import h1_history
    import intraday_history
    from ui.panels import alert_center_panel as panel_module

    created: dict[int, _DelayedCache] = {}

    def cache_factory(interval_minutes=60, **kwargs):
        return created.setdefault(int(interval_minutes), _DelayedCache(interval_minutes, **kwargs))

    monkeypatch.setattr(intraday_history, "IntradayHistoryCache", cache_factory)
    monkeypatch.setattr(h1_history, "H1HistoryCache", lambda **kwargs: cache_factory(60, **kwargs))
    monkeypatch.setattr(panel_module, "IntradayHistoryCache", cache_factory, raising=False)
    monkeypatch.setattr(panel_module, "H1HistoryCache", lambda **kwargs: cache_factory(60, **kwargs), raising=False)
    return _panel(monkeypatch, tmp_path), created


def test_delivered_m15_cache_data_is_evaluated_in_the_same_completed_bucket(monkeypatch, tmp_path):
    """A fetch completing after the first empty read must not wait for M30's next bucket."""
    panel, caches = _manual_pullback_panel(monkeypatch, tmp_path)
    panel.arm_chart_watch_for("NVDA", "LONG", WATCH_KIND)
    watch = dataclasses.replace(
        panel._chart_watches[0],
        armed_at=bar_end(M15_RECLAIM_INDEX - 4, 15),
        triggers=(TRIGGER_RECLAIM,),
    )
    panel._chart_watches = [watch]
    moment = bar_end(M15_RECLAIM_INDEX, 15)

    panel._poll_pullback_watches(now=moment)
    settle_pullback(panel)
    assert panel._alerts == []
    assert len(caches[15].requests) == 1

    # This is the cache worker's completed delivery, deliberately after the
    # first evaluation.  No sleep: the next ordinary poll owns the retry.
    caches[15].rows["NVDA"] = make_bars(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)
    panel._poll_pullback_watches(now=moment + timedelta(minutes=1))
    settle_pullback(panel)

    fires = [alert for alert in panel._alerts if alert.symbol == "NVDA"]
    assert len(fires) == 1
    assert len(caches[15].requests) == 1
    # A further tick with unchanged delivered data is quiet: no duplicate
    # event, request, timer or evaluation worker.
    panel._poll_pullback_watches(now=moment + timedelta(minutes=2))
    settle_pullback(panel)
    assert len([alert for alert in panel._alerts if alert.symbol == "NVDA"]) == 1
    assert len(caches[15].requests) == 1


def test_delivered_h1_fallback_is_retried_within_its_same_bucket_and_keeps_pacing(monkeypatch, tmp_path):
    """H1 has the same stale-read problem, without widening its one-watch pacing."""
    panel, caches = _manual_pullback_panel(monkeypatch, tmp_path)
    panel.PULLBACK_H1_BATCH_LIMIT = 1
    panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND)
    watch = dataclasses.replace(
        panel._chart_watches[0],
        armed_at=GOLDEN_CONFIRM_DT - timedelta(days=1),
        triggers=(TRIGGER_H1,),
    )
    panel._chart_watches = [watch]
    panel._m5_bars_for = lambda *_args, **_kwargs: []
    moment = GOLDEN_CONFIRM_DT + timedelta(hours=1)

    panel._poll_pullback_watches(now=moment)
    assert panel._alerts == []
    initial_requests = len(caches[60].requests)
    assert initial_requests >= 1

    caches[60].rows["AAPL"] = golden_long_h1_bars()[0]
    panel._poll_pullback_watches(now=moment + timedelta(minutes=1))

    fires = [alert for alert in panel._alerts if alert.symbol == "AAPL"]
    assert len(fires) == 1
    assert len(caches[60].requests) == initial_requests


def test_watch_fired_rows_keep_the_measured_sma_lrsi_and_h1_provenance(monkeypatch, tmp_path):
    """Future timing disputes can replay the event from its existing review row."""
    from chart_watch import ChartWatch, ChartWatchTrigger

    panel, _caches = _manual_pullback_panel(monkeypatch, tmp_path)
    panel._push_armed_watch = lambda _hit: None
    watch = ChartWatch(
        symbol="NVDA",
        kind=WATCH_KIND,
        armed_at=bar_end(M15_RECLAIM_INDEX - 4, 15),
        side="LONG",
        watch_id="ar1-provenance",
    )
    panel._record_pullback_fires(
        [
            ChartWatchTrigger(
                watch=watch,
                price=129.8,
                bar_dt=bar_end(M15_RECLAIM_INDEX, 15),
                message="NVDA LONG: Pullback - M30 SMA held, then M15 crossed",
                resolved_side="LONG",
                details={
                    "trigger": "reclaim_then_lrsi",
                    "timeframe": "M30",
                    "rule_version": "pullback_sma_reclaim_v1",
                    "lrsi_from_below_50": True,
                    "bar_dt": bar_end(M15_RECLAIM_INDEX, 15).isoformat(),
                    "sma": 117.34933333333335,
                    "close": 118.7,
                    "lrsi": 86.81175688256107,
                    "atr": 1.5806095172134358,
                    "cross_timeframe": "M15",
                    "cross_bar_dt": bar_end(M15_RECLAIM_INDEX, 15).isoformat(),
                    "cross_lrsi": 86.81175688256107,
                    "sma_bar_dt": bar_end(M15_RECLAIM_INDEX, 15).isoformat(),
                },
            ),
            ChartWatchTrigger(
                watch=watch,
                price=112.0,
                bar_dt=GOLDEN_CONFIRM_DT,
                message="NVDA LONG: Pullback - H1 15-EMA bounce",
                resolved_side="LONG",
                details={
                    "trigger": TRIGGER_H1,
                    "timeframe": "H1",
                    "rule_version": "h1_ema_bounce_v1",
                    "touch_bar_dt": GOLDEN_CONFIRM_DT.isoformat(),
                    "confirm_bar_dt": GOLDEN_CONFIRM_DT.isoformat(),
                    "ema": 111.531477286941,
                    "atr": 1.0,
                    "distance_atr": 0.02,
                    "skipped_bars": 0,
                },
            ),
        ],
        GOLDEN_CONFIRM_DT,
    )

    rows = _events(tmp_path, "watch_fired")
    assert len(rows) == 2
    sma_detail = rows[0]["detail"]
    h1_detail = rows[1]["detail"]
    assert {"bar_dt", "sma", "close", "lrsi", "atr", "cross_timeframe", "cross_bar_dt", "cross_lrsi", "sma_bar_dt"} <= set(sma_detail)
    assert {"touch_bar_dt", "confirm_bar_dt", "ema", "atr", "distance_atr", "skipped_bars"} <= set(h1_detail)
    assert "bars" not in sma_detail and "bars" not in h1_detail
