"""PCT-1 review round - the six blockers, against the REAL seams.

Every test here was red on 508f44cd, the tip the reviewer measured. What each
one holds:

* **B1** - `price_alert_service` de-duplicated `notify_armed_watch` by
  `watch_id` for the life of the process, which was right while an armed watch
  fired once and disarmed. A Pullback alert is a STANDING arm, so only its
  FIRST fire ever reached the phone. The push now carries an `event_key`. The
  tester's `_PushRecorder` hid this, so this file drives the real service with
  only `_deliver_armed_watch` stubbed.
* **B2** - the auto-arm wrote `arm_watch`, which `review_learning.TAKE_ACTIONS`
  scores as a trader TAKE: 95 of them on the first tick.
* **B3** - 1.92 s on the Qt thread on the first tick. The evaluation is now
  bucket-gated and runs off-thread.
* **B6** - a declined auto watch could never be turned back on, and the arm
  button read ARMED beside an empty board.
* advisory 1 - `ChartWatch.fired` keyed on the trigger alone lost one of the
  two timeframes across a save.
* advisory 5 - ~85 fires a session at 108 watches, 70 % of them `sma_retest`.
  An auto-armed watch writes the retest without a push; a hand-armed one
  pushes everything.
"""

from __future__ import annotations

import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from test_pct1_pullback_alert import (  # noqa: E402
    M15_LONG_CLOSES,
    M15_RECLAIM_INDEX,
    bar_dt,
    bar_end,
    make_bars,
)
from test_pct1_pullback_desk import (  # noqa: E402
    WATCH_KIND,
    _claim,
    _events,
    _install_stub_caches,
    _panel,
    _pullback_watches,
    _qt_app,
    settle_pullback,
)

CLAIM_SOURCE_TEXT = "auto: claimed pick"


# ---------------------------------------------------------------------------
# B1 - the real push door, with only the transport stubbed
# ---------------------------------------------------------------------------
def _real_service(monkeypatch):
    """A real `PriceAlertService` whose DELIVERY is a list, in DESK mode.

    Its 60-second poll timer is stopped at birth: this file is about the
    armed-watch door, and a live timer left behind in the shared QApplication
    would read `price_alerts` inside whatever test ran next.
    """
    import autopilot_core
    from ui.services.price_alert_service import PriceAlertService

    monkeypatch.setattr(autopilot_core, "read_auto_pilot_mode", lambda *a, **k: "DESK")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        PriceAlertService,
        "_deliver_armed_watch",
        lambda self, title, message: sent.append((title, message)),
    )
    service = PriceAlertService()
    service._timer.stop()
    return service, sent


def test_a_standing_arm_reaches_the_phone_on_every_new_bar(monkeypatch):
    """Three fires on ONE watch are three buzzes; the same bar twice is one.

    The de-duplication key is the EVENT - watch, trigger, timeframe, bar - and
    a caller that names no key keeps the old watch-id behaviour, which is what
    every one-shot arm on this desk relies on.
    """
    _qt_app()
    service, sent = _real_service(monkeypatch)
    try:
        first = service.notify_armed_watch(
            watch_id="pct1-standing",
            title="Pullback alert: NVDA",
            message="NVDA LONG: Pullback - M15 150-SMA reclaim",
            event_key="pct1-standing:sma_reclaim_lrsi:M15:2026-08-27T08:00:00",
        )
        again = service.notify_armed_watch(
            watch_id="pct1-standing",
            title="Pullback alert: NVDA",
            message="NVDA LONG: Pullback - M15 150-SMA reclaim",
            event_key="pct1-standing:sma_reclaim_lrsi:M15:2026-08-27T08:00:00",
        )
        later_bar = service.notify_armed_watch(
            watch_id="pct1-standing",
            title="Pullback alert: NVDA",
            message="NVDA LONG: Pullback - M15 150-SMA retest held",
            event_key="pct1-standing:sma_retest:M15:2026-08-27T08:15:00",
        )
        other_timeframe = service.notify_armed_watch(
            watch_id="pct1-standing",
            title="Pullback alert: NVDA",
            message="NVDA LONG: Pullback - M30 75-SMA reclaim",
            event_key="pct1-standing:sma_reclaim_lrsi:M30:2026-08-27T08:30:00",
        )
    finally:
        service.shutdown()

    assert bool(first.get("ok")) is True
    assert again.get("deduplicated") is True
    assert bool(later_bar.get("ok")) is True
    assert bool(other_timeframe.get("ok")) is True
    assert len(sent) == 3


def test_a_one_shot_arm_still_buzzes_once_per_watch_id(monkeypatch):
    """Every caller written before PCT-1 passes no key and must not change."""
    _qt_app()
    service, sent = _real_service(monkeypatch)
    try:
        first = service.notify_armed_watch(
            watch_id="ws10c-1", title="H1", message="AAPL LONG"
        )
        again = service.notify_armed_watch(
            watch_id="ws10c-1", title="H1", message="AAPL LONG"
        )
        other = service.notify_armed_watch(
            watch_id="ws10c-2", title="H1", message="MSFT SHORT"
        )
    finally:
        service.shutdown()

    assert bool(first.get("ok")) is True
    assert again.get("deduplicated") is True
    assert bool(other.get("ok")) is True
    assert len(sent) == 2


# ---------------------------------------------------------------------------
# B2 - a machine arm is not a trader take
# ---------------------------------------------------------------------------
def test_the_auto_arm_rows_are_never_scored_as_the_traders_takes():
    import review_learning

    for action in ("auto_arm_watch", "watch_retired_source_gone"):
        assert review_learning._is_take({"action": action}) is False
        assert action not in review_learning.TAKE_ACTIONS
        assert action not in review_learning.REJECT_ACTIONS
    # The HAND-armed button is still a take; that is the whole distinction.
    assert review_learning._is_take({"action": "arm_watch"}) is True

    rows = [
        {"action": "auto_arm_watch", "detail": {"kind": WATCH_KIND}},
        {"action": "watch_retired_source_gone", "detail": {"kind": WATCH_KIND}},
        {"action": "arm_watch", "detail": {"kind": WATCH_KIND}},
    ]
    conversion = review_learning.watch_conversion(rows)
    assert conversion["kinds"][WATCH_KIND] == {"armed": 1}


@pytest.mark.skip(reason="Pullback alerts are manual-only (trader 2026-09-17).")
def test_the_sweep_writes_auto_arm_watch_and_never_arm_watch(monkeypatch, tmp_path):
    _install_stub_caches(monkeypatch, bars={})
    panel = _panel(monkeypatch, tmp_path)
    _claim(tmp_path, "NVDA", "LONG")

    panel._poll_pullback_watches(now=datetime.now())
    settle_pullback(panel)

    assert [row["symbol"] for row in _events(tmp_path, "auto_arm_watch")] == ["NVDA"]
    assert _events(tmp_path, "arm_watch") == []
    detail = dict(_events(tmp_path, "auto_arm_watch")[0].get("detail") or {})
    assert detail.get("auto") is True
    assert detail.get("source_text") == CLAIM_SOURCE_TEXT


def test_the_hand_armed_button_still_writes_arm_watch(monkeypatch, tmp_path):
    _install_stub_caches(monkeypatch, bars={})
    panel = _panel(monkeypatch, tmp_path)

    assert panel.arm_chart_watch_for("AAPL", "LONG", WATCH_KIND) is True

    assert [row["symbol"] for row in _events(tmp_path, "arm_watch")] == ["AAPL"]
    assert _events(tmp_path, "auto_arm_watch") == []


# ---------------------------------------------------------------------------
# B3 - the Qt thread
# ---------------------------------------------------------------------------
def _ninety_five_claims(tmp_path) -> list[str]:
    symbols = [f"SYM{index:03d}" for index in range(95)]
    for symbol in symbols:
        _claim(tmp_path, symbol, "LONG")
    return symbols


@pytest.mark.skip(reason="Pullback alerts are manual-only (trader 2026-09-17).")
def test_a_tick_with_ninety_five_warm_watches_is_under_fifty_milliseconds(
    monkeypatch, tmp_path
):
    """The reviewer measured 1.92 s on the first tick and ~0.7 s every minute
    after. Measured with a monotonic clock around the slot itself; the worker
    is joined AFTER the measurement, so nothing it does is counted and nothing
    is faked."""
    m15 = make_bars(M15_LONG_CLOSES, 15)
    bars = {(symbol, 15): m15 for symbol in (f"SYM{index:03d}" for index in range(95))}
    _install_stub_caches(monkeypatch, bars=bars)
    panel = _panel(monkeypatch, tmp_path)
    _ninety_five_claims(tmp_path)

    # Tick one arms them all; tick two is the steady state the trader lives in.
    first = time.monotonic()
    panel._poll_pullback_watches(now=bar_end(M15_RECLAIM_INDEX, 15))
    first_seconds = time.monotonic() - first
    settle_pullback(panel, timeout=20.0)
    assert len(_pullback_watches(panel)) == 95

    second = time.monotonic()
    panel._poll_pullback_watches(
        now=bar_end(M15_RECLAIM_INDEX, 15) + timedelta(seconds=60)
    )
    second_seconds = time.monotonic() - second
    settle_pullback(panel, timeout=20.0)

    print(
        "\nPCT-1 Qt thread, 95 armed watches: arming tick %.1f ms, "
        "warm tick %.1f ms" % (first_seconds * 1000.0, second_seconds * 1000.0)
    )
    assert second_seconds < 0.05, (
        "a warm tick with 95 armed watches held the Qt thread for %.3fs"
        % second_seconds
    )
    # The first tick arms 95 watches and has real work to do; it is still an
    # order of magnitude off the 1.92 s the reviewer measured.
    assert first_seconds < 0.5, "the arming tick took %.3fs" % first_seconds


def test_no_new_bucket_means_no_evaluation_at_all(monkeypatch, tmp_path):
    """A completed bar is the only thing that can change the answer."""
    m15 = make_bars(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)
    _install_stub_caches(monkeypatch, bars={("NVDA", 15): m15})
    panel = _panel(monkeypatch, tmp_path)
    assert panel.arm_chart_watch_for("NVDA", "LONG", WATCH_KIND) is True
    moment = bar_end(M15_RECLAIM_INDEX, 15)

    assert panel._dispatch_pullback_sma_evaluation(panel._chart_watches, moment) is True
    settle_pullback(panel)

    # Same completed bucket a minute later: nothing to ask.
    assert (
        panel._dispatch_pullback_sma_evaluation(
            panel._chart_watches, moment + timedelta(minutes=1)
        )
        is False
    )
    # A new quarter hour closes and the question is live again.
    assert (
        panel._dispatch_pullback_sma_evaluation(
            panel._chart_watches, moment + timedelta(minutes=15)
        )
        is True
    )
    settle_pullback(panel)


def test_the_sma_evaluation_never_runs_on_the_qt_thread(monkeypatch, tmp_path):
    from PySide6.QtCore import QThread

    app = _qt_app()
    m15 = make_bars(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)
    _install_stub_caches(monkeypatch, bars={("NVDA", 15): m15})
    panel = _panel(monkeypatch, tmp_path)
    seen: list[bool] = []
    original = panel._evaluate_one_pullback_job

    def _spy(job, moment):
        seen.append(QThread.currentThread() == app.thread())
        return original(job, moment)

    monkeypatch.setattr(panel, "_evaluate_one_pullback_job", _spy)
    panel.arm_chart_watch_for("NVDA", "LONG", WATCH_KIND)
    # `arm_chart_watch_for` stamps `armed_at = now()`, i.e. today, which is
    # after the 2026-08 golden bars; the post-arm fence would then correctly
    # refuse. The same pin the tester's own fire test uses.
    import dataclasses

    panel._chart_watches = [
        dataclasses.replace(watch, armed_at=bar_dt(M15_RECLAIM_INDEX - 9, 15))
        for watch in panel._chart_watches
    ]

    panel._poll_pullback_watches(now=bar_end(M15_RECLAIM_INDEX, 15))
    settle_pullback(panel)

    assert seen == [False], seen
    assert len(_events(tmp_path, "watch_fired")) == 1


# ---------------------------------------------------------------------------
# B6 - a declined watch can be turned back on
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="Pullback alerts are manual-only (trader 2026-09-17).")
def test_a_declined_watch_reads_as_not_armed_and_can_be_re_armed(
    monkeypatch, tmp_path
):
    _install_stub_caches(monkeypatch, bars={})
    panel = _panel(monkeypatch, tmp_path)
    _claim(tmp_path, "NVDA", "LONG")
    panel._poll_pullback_watches(now=datetime.now())
    settle_pullback(panel)
    assert WATCH_KIND in panel.armed_watch_kinds("NVDA")

    assert panel.disarm_chart_watch_for("NVDA", WATCH_KIND) is True

    # The button must not read ARMED beside an empty Armed board.
    assert WATCH_KIND not in panel.armed_watch_kinds("NVDA")
    assert _pullback_watches(panel) == []

    # And the chart can turn it back on, as a HAND-armed watch.
    assert panel.arm_chart_watch_for("NVDA", "LONG", WATCH_KIND, source_text="chart") is True
    back = _pullback_watches(panel, "NVDA")
    assert len(back) == 1
    assert bool(back[0].declined) is False
    assert back[0].source_text == "chart"
    assert panel._is_auto_pullback_watch(back[0]) is False

    # The sweep leaves it alone from here on: it is the trader's now.
    panel._poll_pullback_watches(now=datetime.now() + timedelta(minutes=1))
    settle_pullback(panel)
    still = _pullback_watches(panel, "NVDA")
    assert len(still) == 1
    assert still[0].watch_id == back[0].watch_id
    assert bool(still[0].declined) is False


# ---------------------------------------------------------------------------
# advisory 1 - both timeframes survive a save
# ---------------------------------------------------------------------------
def test_the_fired_stamp_names_the_timeframe_as_well_as_the_trigger(tmp_path):
    import dataclasses as _dc

    from chart_watch import ChartWatch, load_chart_watches, save_chart_watches
    from ui.panels.alert_center_panel import AlertCenterPanel

    assert AlertCenterPanel.pullback_fire_key("sma_reclaim_lrsi", "M15") == (
        "sma_reclaim_lrsi@M15"
    )
    watch = _dc.replace(
        ChartWatch(
            symbol="NVDA",
            kind=WATCH_KIND,
            armed_at=datetime(2026, 9, 14, 7, 15),
            side="LONG",
            watch_id="pct1-two-timeframes",
        ),
        fired={
            AlertCenterPanel.pullback_fire_key("sma_reclaim_lrsi", "M15"): "a",
            AlertCenterPanel.pullback_fire_key("sma_reclaim_lrsi", "M30"): "b",
        },
    )
    path = tmp_path / "chart_watches.json"
    save_chart_watches([watch], path, market_date="2026-09-14")

    back = load_chart_watches(path, market_date="2026-09-15")

    assert dict(back[0].fired) == {
        "sma_reclaim_lrsi@M15": "a",
        "sma_reclaim_lrsi@M30": "b",
    }


# ---------------------------------------------------------------------------
# advisory 5 - the phone volume split
# ---------------------------------------------------------------------------
def _retest_bars():
    """The M15 golden plus one bar that retests the SMA and holds it."""
    from indicators.atr import wilder_atr

    from test_pct1_pullback_alert import (
        HALF_RANGE,
        M15_RETEST_CLOSE,
        M15_RETEST_TAG_LOW,
    )

    closes = list(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1]) + [M15_RETEST_CLOSE]
    bars = make_bars(closes, 15)
    bars[-1] = {
        "dt": bar_dt(M15_RECLAIM_INDEX + 1, 15),
        "open": M15_RETEST_CLOSE,
        "high": M15_RETEST_CLOSE + HALF_RANGE,
        "low": M15_RETEST_TAG_LOW,
        "close": M15_RETEST_CLOSE,
        "volume": 2_000,
    }
    assert wilder_atr(bars, 14)  # the fixture is measurable
    return bars


def _fire_once(monkeypatch, tmp_path, *, auto: bool):
    bars = _retest_bars()
    _install_stub_caches(monkeypatch, bars={("NVDA", 15): bars})
    panel = _panel(monkeypatch, tmp_path)
    service, sent = _real_service(monkeypatch)
    panel.price_alert_service = service
    if auto:
        _claim(tmp_path, "NVDA", "LONG")
        panel._poll_pullback_watches(now=datetime(2026, 8, 17, 6, 30))
        settle_pullback(panel)
        watch = _pullback_watches(panel, "NVDA")[0]
        assert panel._is_auto_pullback_watch(watch) is True
    else:
        panel.arm_chart_watch_for("NVDA", "LONG", WATCH_KIND, source_text="chart")
    import dataclasses

    panel._chart_watches = [
        dataclasses.replace(watch, armed_at=bar_dt(M15_RECLAIM_INDEX - 9, 15))
        if watch.kind == WATCH_KIND
        else watch
        for watch in panel._chart_watches
    ]
    panel._pullback_judged = {}
    try:
        panel._poll_pullback_watches(now=bar_end(M15_RECLAIM_INDEX + 1, 15))
        settle_pullback(panel)
    finally:
        service.shutdown()
    rows = _events(tmp_path, "watch_fired")
    return rows, sent


@pytest.mark.skip(reason="Pullback alerts are manual-only (trader 2026-09-17).")
def test_an_auto_armed_retest_is_recorded_and_drawn_but_never_pushed(
    monkeypatch, tmp_path
):
    rows, sent = _fire_once(monkeypatch, tmp_path, auto=True)

    triggers = [str((row.get("detail") or {}).get("trigger") or "") for row in rows]
    assert "sma_retest" in triggers, triggers
    # Every fire is recorded; only the retest is kept off the phone.
    pushed = [title for title, _message in sent]
    assert len(sent) == len([name for name in triggers if name != "sma_retest"])
    assert all("NVDA" in title for title in pushed)


def test_a_hand_armed_retest_still_buzzes(monkeypatch, tmp_path):
    rows, sent = _fire_once(monkeypatch, tmp_path, auto=False)

    triggers = [str((row.get("detail") or {}).get("trigger") or "") for row in rows]
    assert "sma_retest" in triggers, triggers
    assert len(sent) == len(triggers)


@pytest.mark.skip(reason="Pullback alerts are manual-only (trader 2026-09-17).")
def test_one_arming_tick_with_ninety_five_watches_is_a_handful_of_downloads(
    monkeypatch, tmp_path
):
    """The reviewer counted 285 single-ticker downloads and 286 threads.

    Driven through the panel with the REAL `IntradayHistoryCache` (only the
    download is a list), so this is the end-to-end number rather than the
    cache's own: 95 symbols on each of M15 and M30 is two chunks apiece.
    """
    import intraday_history

    calls: list[tuple[int, int]] = []
    real = intraday_history.IntradayHistoryCache

    class _Counting(real):
        def __init__(self, interval_minutes=60, **kwargs):
            kwargs["downloader"] = self._download
            super().__init__(interval_minutes, **kwargs)

        def _download(self, symbols, **_kwargs):
            calls.append((self.interval_minutes, len(symbols)))
            return {name: None for name in symbols}

    monkeypatch.setattr(intraday_history, "IntradayHistoryCache", _Counting)
    import h1_history

    monkeypatch.setattr(h1_history, "H1HistoryCache", lambda **kw: _Counting(60))
    panel = _panel(monkeypatch, tmp_path)
    _ninety_five_claims(tmp_path)

    panel._poll_pullback_watches(now=bar_end(M15_RECLAIM_INDEX, 15))
    settle_pullback(panel, timeout=20.0)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not [
            thread
            for thread in threading.enumerate()
            if thread.is_alive() and "history" in thread.name
        ]:
            break
        time.sleep(0.01)

    assert len(_pullback_watches(panel)) == 95
    print("\nPCT-1 downloads for one arming tick of 95 watches: %d" % len(calls))
    # Two chunks of 50 for each of the two SMA timeframes, and the paced H1
    # leg's dozen in one or two more. Measured: 6. The bound is loose because
    # where the batching window happens to fall can split one chunk; what is
    # NOT allowed to drift is the shape - a chunk is at most 50 names, and 95
    # watches are single figures of requests rather than 285.
    assert len(calls) <= 10, calls
    assert max(size for _interval, size in calls) <= 50
    assert sum(size for _interval, size in calls) <= 95 * 3


def test_the_worker_thread_is_the_only_one_and_it_finishes(monkeypatch, tmp_path):
    """No thread leak: one evaluation at a time, and it exits."""
    m15 = make_bars(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)
    _install_stub_caches(monkeypatch, bars={("NVDA", 15): m15})
    panel = _panel(monkeypatch, tmp_path)
    panel.arm_chart_watch_for("NVDA", "LONG", WATCH_KIND)

    for minute in range(4):
        panel._poll_pullback_watches(
            now=bar_end(M15_RECLAIM_INDEX, 15) + timedelta(minutes=15 * minute)
        )
        assert (
            len(
                [
                    thread
                    for thread in threading.enumerate()
                    if thread.is_alive() and thread.name == "pullback-eval"
                ]
            )
            <= 1
        )
        settle_pullback(panel)

    assert [
        thread
        for thread in threading.enumerate()
        if thread.is_alive() and thread.name == "pullback-eval"
    ] == []
