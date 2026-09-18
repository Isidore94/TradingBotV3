"""PCT-1 - the Pullback alert on the desk: cache, kind, arm bar, poll, claims.

The other half of `tests/test_pct1_pullback_alert.py` (the pure rule), written
RED before any of it exists. `docs/PULLBACK_COMPRESSION_TRENDLINE_PLAN.md`
section 5, items 2-7.

What is driven here, and through which real seam:

* ``scripts/intraday_history.py`` ``IntradayHistoryCache(15)`` - the H1 cache's
  bucket math parameterised by ``interval_minutes``, with a FAKE downloader, so
  a quarter-hour cache asks once per completed quarter hour and the 60-minute
  instance still keeps the session-aligned H1 cadence the RV-H1-HISTORY repair
  pinned. No yfinance import anywhere in this file.
* ``scripts/chart_watch.py`` - the ``pullback`` kind, ``ChartWatch.triggers`` /
  ``fired`` / ``declined``, and a STORED ``h1_ema_bounce`` row loading as a
  ``pullback`` watch that carries only the H1 trigger (nothing the trader armed
  is lost by the rename).
* ``ui/widgets/arm_bar.py`` offscreen - one "Pullback alert" button and no
  "H1 retester" one.
* ``AlertCenterPanel._poll_pullback_watches`` - the auto-arm from
  ``claimed_picks.active_claims()`` and ``FocusPickStore.all_focus("swing")``,
  the ``declined`` memory, and ONE ``watch_fired`` row + ONE push per fire.
* the capture rail - the three new claim ids (the mirror of
  ``tests/test_qt_alert_capture.py::test_the_rail_offers_every_claim_the_trader_asked_for``,
  which this file deliberately does not edit).
"""

from __future__ import annotations

import dataclasses
import json
import os
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
    RULE_VERSION,
    TRIGGER_H1,
    TRIGGER_RECLAIM,
    TRIGGER_RETEST,
    TRIGGER_THEN_LRSI,
    bar_dt,
    bar_end,
    make_bars,
)

WATCH_KIND = "pullback"
BUTTON_LABEL = "Pullback alert"
RETIRED_BUTTON_LABEL = "H1 retester"
ALL_TRIGGERS = {TRIGGER_H1, TRIGGER_RECLAIM, TRIGGER_THEN_LRSI, TRIGGER_RETEST}

CLAIM_SOURCE_TEXT = "auto: claimed pick"
FOCUS_SOURCE_TEXT = "auto: swing Focus"

#: The three claim names, exactly as the trader asked for them.
NEW_CLAIM_IDS = ("pullback_sma_reclaim", "trendline_break", "compression_break")
NEW_CLAIM_GROUP = "Entry timing and breaks"
NEW_CLAIM_LABELS = {
    "pullback_sma_reclaim": "Pullback reclaim (M15 150 / M30 75 SMA)",
    "trendline_break": "Trendline break",
    "compression_break": "Compression break",
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def _qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ModuleNotFoundError:  # pragma: no cover - headless without Qt
        pytest.skip("PySide6 not installed")
    return QApplication.instance() or QApplication([])


class _Frame:
    """The shape a yfinance-reading frame parser sees, from plain bar dicts."""

    columns = None

    def __init__(self, bars):
        self._bars = list(bars or ())
        self.empty = not self._bars

    def iterrows(self):
        for bar in self._bars:
            yield bar["dt"], {
                "Open": bar["open"],
                "High": bar["high"],
                "Low": bar["low"],
                "Close": bar["close"],
            }


class _StubCache:
    """A cache stand-in that answers from memory and records every ask.

    Built to the H1 cache's read contract (`bars_for`, `unavailable`,
    `last_refresh_failed`, `request`) plus the interval it was constructed for,
    so one factory can serve the H1, M15 and M30 caches at once and the test
    can see WHICH interval the panel asked for.
    """

    def __init__(self, interval_minutes=60, *, downloader=None, period="1mo", **_kw):
        self.interval_minutes = int(interval_minutes)
        self.requests: list[datetime] = []
        self.built.append(self)

    built: list["_StubCache"] = []
    bars: dict[tuple[str, int], list[dict]] = {}
    failed: set[tuple[str, int]] = set()

    def bars_for(self, symbol):
        key = (str(symbol or "").strip().upper(), self.interval_minutes)
        return [dict(bar) for bar in (self.bars.get(key) or ())]

    def unavailable(self, symbol):
        key = (str(symbol or "").strip().upper(), self.interval_minutes)
        return key in self.failed and not self.bars.get(key)

    def last_refresh_failed(self, symbol):
        key = (str(symbol or "").strip().upper(), self.interval_minutes)
        return key in self.failed

    def request(self, symbol, *, now=None):
        self.requests.append(now)
        return False

    def fetch_now(self, symbol, *, now=None):
        return self.bars_for(symbol)


def _install_stub_caches(monkeypatch, *, bars=None, failed=None):
    """Point every history cache the panel can build at memory, not the network."""
    import h1_history

    _StubCache.built = []
    _StubCache.bars = dict(bars or {})
    _StubCache.failed = set(failed or ())
    try:
        import intraday_history

        monkeypatch.setattr(
            intraday_history, "IntradayHistoryCache", _StubCache, raising=False
        )
    except ModuleNotFoundError:
        pass
    monkeypatch.setattr(h1_history, "H1HistoryCache", _StubCache, raising=False)
    from ui.panels import alert_center_panel as panel_module

    for name in ("IntradayHistoryCache", "H1HistoryCache"):
        if hasattr(panel_module, name):
            monkeypatch.setattr(panel_module, name, _StubCache, raising=False)
    return _StubCache


class _PushRecorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def notify_armed_watch(self, *, watch_id, title, message):
        self.calls.append({"watch_id": watch_id, "title": title, "message": message})
        return {"ok": True}


def _focus_service(tmp_path):
    from focus_picks import FocusPickStore
    from ui.services.focus_service import FocusService

    store = FocusPickStore(
        focus_longs_path=tmp_path / "focus_longs.txt",
        focus_shorts_path=tmp_path / "focus_shorts.txt",
        longs_path=tmp_path / "longs.txt",
        shorts_path=tmp_path / "shorts.txt",
        membership_path=tmp_path / "membership.json",
    )
    return FocusService(store)


def _panel(monkeypatch, tmp_path, *, focus_service=None):
    _qt_app()
    from ui.panels.alert_center_panel import AlertCenterPanel
    from ui.widgets.symbol_snapshot_dialog import SymbolSnapshotWidget

    monkeypatch.setattr(SymbolSnapshotWidget, "set_symbol", lambda *a, **k: None)
    made = AlertCenterPanel(
        focus_service=focus_service,
        ignored_symbols_path=tmp_path / "ignored.json",
        parked_symbols_path=tmp_path / "parked.json",
        chart_watches_path=tmp_path / "chart_watches.json",
        review_events_path=tmp_path / "alert_review_events.jsonl",
        claimed_picks_path=tmp_path / "claimed_picks.jsonl",
    )
    made._chart_watches_path = tmp_path / "chart_watches.json"
    monkeypatch.setattr(made, "_m5_bars_for", lambda symbol, **kw: [])
    monkeypatch.setattr(made, "_d1_bars_for", lambda symbol, **kw: [])
    monkeypatch.setattr(made, "_alerts_may_sound", lambda: False)
    return made


def _events(tmp_path, action: str) -> list[dict]:
    path = tmp_path / "alert_review_events.jsonl"
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row.get("action") == action]


def _pullback_watches(panel, symbol=None):
    return [
        watch
        for watch in panel._chart_watches
        if watch.kind == WATCH_KIND
        and (symbol is None or watch.symbol == symbol)
        and not bool(getattr(watch, "declined", False))
    ]


def settle_pullback(panel, timeout: float = 5.0) -> None:
    """Wait for the off-thread pullback evaluation and deliver its fires.

    HARNESS ONLY, added by the builder in the review round: blocker 3 moved
    the SMA / LRSI / ATR passes off the Qt thread (95 armed watches cost
    1.92 s there), so a fire now arrives on a QUEUED signal a moment after the
    poll returns instead of inside it. This waits for that worker and spins
    the event loop, which is exactly what the desk's own loop does a
    millisecond later. Not one assertion below changes.
    """
    from PySide6.QtWidgets import QApplication

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        working = [
            thread
            for thread in threading.enumerate()
            if thread.is_alive() and thread.name == "pullback-eval"
        ]
        if not working and not getattr(panel, "_pullback_eval_busy", False):
            break
        time.sleep(0.005)
    QApplication.processEvents()


def _settle(prefix: str = "", timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = [
            thread
            for thread in threading.enumerate()
            if thread.is_alive()
            and ("history" in thread.name or thread.name.startswith(prefix or "\0"))
        ]
        if not alive:
            return
        time.sleep(0.01)
    raise AssertionError("a history fetch thread never finished")


# ---------------------------------------------------------------------------
# Item 2 - the intraday history cache
# ---------------------------------------------------------------------------
SESSION_DAY = datetime(2026, 8, 26)  # a Wednesday, a full exchange session
NEXT_SESSION_DAY = datetime(2026, 8, 27)


def _at(day: datetime, hour: int, minute: int = 0) -> datetime:
    return day.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _real_cache(interval_minutes: int, bars, *, on_download=None):
    from intraday_history import IntradayHistoryCache

    rows = list(bars or ())

    def _download(symbol, **kwargs):
        if on_download is not None:
            on_download(symbol, kwargs)
        return _Frame(rows)

    return IntradayHistoryCache(interval_minutes, downloader=_download)


def test_a_fifteen_minute_cache_asks_once_per_completed_quarter_hour():
    """A new answer can only exist when a quarter-hour bucket has CLOSED.

    06:30, 06:45 ... 12:45 market-local. Two asks inside one bucket are one
    question; the clock-hour key the H1 cache started with would refetch four
    times an hour here, and the evening poll would refetch for ever.
    """
    cache = _real_cache(15, [])
    asks = [
        (_at(SESSION_DAY, 11, 47), True),   # the 11:30 bucket closed at 11:45
        (_at(SESSION_DAY, 11, 52), False),  # still the same completed bucket
        (_at(SESSION_DAY, 12, 1), True),    # 11:45 closed at 12:00
        # Lead ruling 2026-09-15, after the PCT-1 review reproduced the cost:
        # this expectation was WRONG as written. Four more buckets close
        # between 12:01 and the bell, and the last of them - 12:45-13:00, the
        # session's closing bar - is new and completed, so the answer here is
        # True. A cache that refused here never fetched the closing bar of any
        # session while its health cell still read `from yfinance`.
        (_at(SESSION_DAY, 13, 30), True),   # the 12:45 bucket closed at the bell
        (_at(NEXT_SESSION_DAY, 6, 40), False),  # 06:30 has not closed yet
        (_at(NEXT_SESSION_DAY, 6, 46), True),   # now it has
    ]

    got = []
    for moment, _expected in asks:
        got.append(cache.request("AAPL", now=moment))
        _settle()

    assert got == [expected for _moment, expected in asks], (
        "requests at %s -> %s"
        % ([moment.strftime("%m-%d %H:%M") for moment, _ in asks], got)
    )


def test_a_thirty_minute_cache_keeps_its_own_half_hour_cadence():
    cache = _real_cache(30, [])
    asks = [
        (_at(SESSION_DAY, 11, 5), True),    # the 10:30 bucket closed at 11:00
        (_at(SESSION_DAY, 11, 25), False),
        (_at(SESSION_DAY, 11, 35), True),   # 11:00 closed at 11:30
    ]

    got = []
    for moment, _expected in asks:
        got.append(cache.request("AAPL", now=moment))
        _settle()

    assert got == [expected for _moment, expected in asks]


def test_the_sixty_minute_instance_still_keeps_the_session_h1_cadence():
    """`h1_history.H1HistoryCache` becomes `IntradayHistoryCache(60)`, so the
    RV-H1-HISTORY cadence (session-aligned buckets, the short 12:30 one closing
    at the bell) has to survive the generalisation unchanged."""
    cache = _real_cache(60, [])
    asks = [
        (_at(SESSION_DAY, 11, 45), True),   # the 10:30 bucket closed at 11:30
        (_at(SESSION_DAY, 12, 15), False),  # same bucket
        (_at(SESSION_DAY, 13, 0), True),    # the short 12:30 bucket, at the bell
        (_at(SESSION_DAY, 16, 0), False),   # the evening asks for nothing
    ]

    got = []
    for moment, _expected in asks:
        got.append(cache.request("AAPL", now=moment))
        _settle()

    assert got == [expected for _moment, expected in asks]


def test_the_fifteen_minute_cache_drops_the_forming_bucket_and_asks_for_15m():
    """Completed bars only, and the request is for regular hours."""
    bars = make_bars(M15_LONG_CLOSES[:40], 15)
    seen: list[dict] = []
    cache = _real_cache(15, bars, on_download=lambda symbol, kw: seen.append(kw))

    # `now` sits one minute inside the LAST bar, so that bucket is still
    # forming and must not come back.
    got = cache.fetch_now("AAPL", now=bar_dt(39, 15) + timedelta(minutes=1))

    assert len(got) == 39
    assert got[-1]["dt"] == bar_dt(38, 15)
    assert seen and str(seen[0].get("interval")) == "15m"


def test_the_h1_history_module_keeps_its_name_as_the_sixty_minute_cache():
    import h1_history
    from intraday_history import IntradayHistoryCache

    cache = h1_history.H1HistoryCache(downloader=lambda *a, **k: _Frame([]))
    assert isinstance(cache, IntradayHistoryCache)
    assert getattr(cache, "interval_minutes", None) == 60


# ---------------------------------------------------------------------------
# Item 3 - the watch kind
# ---------------------------------------------------------------------------
def test_the_kind_is_one_button_called_pullback_alert_and_it_is_persistent():
    from chart_watch import PERSISTENT_WATCH_KINDS, WATCH_KINDS

    assert WATCH_KINDS[WATCH_KIND] == BUTTON_LABEL
    assert WATCH_KIND in PERSISTENT_WATCH_KINDS


def test_a_new_pullback_watch_carries_all_four_triggers_and_nothing_fired():
    from chart_watch import arm_chart_watch

    watch = arm_chart_watch(
        WATCH_KIND, "nvda", "LONG", [], now=datetime(2026, 8, 26, 7, 15)
    )

    assert set(watch.triggers) == ALL_TRIGGERS
    assert len(watch.triggers) == 4
    assert dict(watch.fired) == {}
    assert bool(getattr(watch, "declined", False)) is False
    assert watch.watch_id


def test_the_reason_names_all_three_families_of_trigger():
    from chart_watch import watch_reason

    reason = watch_reason(WATCH_KIND, "LONG")

    assert reason == (
        "waiting for a pullback entry (LONG): H1 15-EMA bounce, "
        "M15/M30 SMA reclaim + LRSI, SMA retest"
    )
    assert "(SHORT)" in watch_reason(WATCH_KIND, "SHORT")


def test_a_stored_h1_retester_loads_as_a_pullback_watch_with_the_h1_trigger():
    """Nothing the trader armed before the rename is lost, and nothing they did
    not ask for is added to it: the old row becomes a `pullback` watch whose
    ONLY trigger is the H1 bounce it was armed for."""
    from chart_watch import chart_watch_from_dict

    watch = chart_watch_from_dict(
        {
            "symbol": "AAPL",
            "kind": "h1_ema_bounce",
            "armed_at": "2026-09-14T07:15:00",
            "side": "LONG",
            "baseline": None,
            "source_text": "chart",
            "watch_id": "ws10c-legacy",
            "reason": "waiting for an H1 15-EMA bounce (LONG)",
        }
    )

    assert watch is not None
    assert watch.kind == WATCH_KIND
    assert tuple(watch.triggers) == (TRIGGER_H1,)
    assert watch.watch_id == "ws10c-legacy"
    assert watch.armed_at == datetime(2026, 9, 14, 7, 15)


def test_triggers_fired_and_declined_survive_a_save_and_a_reload(tmp_path):
    import dataclasses as _dc

    from chart_watch import ChartWatch, load_chart_watches, save_chart_watches

    watch = ChartWatch(
        symbol="NVDA",
        kind=WATCH_KIND,
        armed_at=datetime(2026, 9, 14, 7, 15),
        side="LONG",
        watch_id="pct1-round-trip",
    )
    watch = _dc.replace(
        watch,
        triggers=(TRIGGER_RECLAIM, TRIGGER_RETEST),
        fired={TRIGGER_RECLAIM: "2026-09-14T08:00:00"},
        declined=True,
    )
    path = tmp_path / "chart_watches.json"
    save_chart_watches([watch], path, market_date="2026-09-14")

    # A different market date: the persistent kind still comes back.
    back = load_chart_watches(path, market_date="2026-09-15")

    assert len(back) == 1
    assert tuple(back[0].triggers) == (TRIGGER_RECLAIM, TRIGGER_RETEST)
    assert dict(back[0].fired) == {TRIGGER_RECLAIM: "2026-09-14T08:00:00"}
    assert bool(back[0].declined) is True


def test_the_arm_bar_shows_one_pullback_button_and_no_h1_retester():
    _qt_app()
    from ui.widgets.arm_bar import ArmBar

    bar = ArmBar()
    bar.set_enabled_for_symbol(True)
    labels = {button.text() for button in bar.watch_buttons.values()}

    assert WATCH_KIND in bar.watch_buttons
    assert any(BUTTON_LABEL in label for label in labels)
    assert not any(RETIRED_BUTTON_LABEL in label for label in labels), labels

    emitted: list[str] = []
    bar.watchToggled.connect(emitted.append)
    bar.watch_buttons[WATCH_KIND].click()
    assert emitted == [WATCH_KIND]


# ---------------------------------------------------------------------------
# Items 4 and 6 - the poll, the fire, the health cell
# ---------------------------------------------------------------------------
def _arm_before(panel, symbol, side, *, armed_at):
    assert panel.arm_chart_watch_for(symbol, side, WATCH_KIND) is True, (
        f"{symbol}: the {BUTTON_LABEL} watch did not arm"
    )
    panel._chart_watches = [
        dataclasses.replace(watch, armed_at=armed_at)
        if watch.kind == WATCH_KIND and watch.symbol == symbol
        else watch
        for watch in panel._chart_watches
    ]
    return next(
        watch
        for watch in panel._chart_watches
        if watch.kind == WATCH_KIND and watch.symbol == symbol
    )


def test_a_m15_reclaim_writes_one_watch_fired_row_and_buzzes_the_phone_once(
    monkeypatch, tmp_path
):
    m15 = make_bars(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)
    _install_stub_caches(monkeypatch, bars={("NVDA", 15): m15})
    panel = _panel(monkeypatch, tmp_path)
    recorder = _PushRecorder()
    panel.price_alert_service = recorder
    _arm_before(panel, "NVDA", "LONG", armed_at=bar_dt(M15_RECLAIM_INDEX - 9, 15))
    before = len(panel._alerts)

    panel._poll_pullback_watches(now=bar_end(M15_RECLAIM_INDEX, 15))
    settle_pullback(panel)

    rows = _events(tmp_path, "watch_fired")
    assert len(rows) == 1, rows
    detail = dict(rows[0].get("detail") or {})
    assert detail.get("trigger") == TRIGGER_RECLAIM
    assert detail.get("timeframe") == "M15"
    assert detail.get("rule_version") == RULE_VERSION
    assert detail.get("lrsi_from_below_50") is True
    assert rows[0]["symbol"] == "NVDA"

    assert len(recorder.calls) == 1
    assert len(panel._alerts) == before + 1
    message = str(panel._alerts[0].trigger)
    assert "M15" in message and "Pullback" in message

    # A pullback watch is a standing arm: one trigger firing does not retire it.
    assert _pullback_watches(panel, "NVDA")


def test_the_same_reclaim_on_the_next_poll_does_not_fire_twice(
    monkeypatch, tmp_path
):
    m15 = make_bars(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)
    _install_stub_caches(monkeypatch, bars={("NVDA", 15): m15})
    panel = _panel(monkeypatch, tmp_path)
    panel.price_alert_service = _PushRecorder()
    _arm_before(panel, "NVDA", "LONG", armed_at=bar_dt(M15_RECLAIM_INDEX - 9, 15))

    panel._poll_pullback_watches(now=bar_end(M15_RECLAIM_INDEX, 15))
    settle_pullback(panel)
    panel._poll_pullback_watches(
        now=bar_end(M15_RECLAIM_INDEX, 15) + timedelta(minutes=1)
    )
    settle_pullback(panel)

    assert len(_events(tmp_path, "watch_fired")) == 1
    assert len(panel.price_alert_service.calls) == 1


def test_the_health_cell_reports_each_timeframe_on_its_own(monkeypatch, tmp_path):
    """`not measured (N of 160 M15 bars)` - an armed surface never says `ok`
    beside a watch that cannot evaluate at all."""
    _install_stub_caches(monkeypatch, bars={})
    panel = _panel(monkeypatch, tmp_path)
    watch = _arm_before(panel, "NVDA", "LONG", armed_at=datetime(2026, 9, 14, 7, 15))

    note = panel._armed_watch_note(watch)

    assert "not measured (0 of 160 M15 bars)" in note
    assert "not measured (0 of 85 M30 bars)" in note
    assert "H1" in note  # the existing H1 state is unchanged, not replaced
    assert note.count(";") == 2, note

    # With a full quarter-hour history the same cell says where it came from.
    # `_StubCache.bars` is a CLASS attribute, so every cache the panel already
    # built starts answering with these bars - no cache is rebuilt and no
    # private attribute name is assumed.
    _StubCache.bars = {
        ("NVDA", 15): make_bars(M15_LONG_CLOSES[: M15_RECLAIM_INDEX + 1], 15)
    }
    filled = panel._armed_watch_note(watch)
    assert "M15 from yfinance" in filled


# ---------------------------------------------------------------------------
# Item 5 - auto-arm from the trader's own picks
# ---------------------------------------------------------------------------
def _claim(tmp_path, symbol, side, *, horizon="d1", now=None):
    import claimed_picks

    return claimed_picks.record_claim(
        symbol=symbol,
        side=side,
        horizon=horizon,
        claimed_setup_id="pullback_sma_reclaim",
        source="test",
        now=now or datetime.now(),
        path=tmp_path / "claimed_picks.jsonl",
    )


def test_the_poll_never_auto_arms_a_pullback_from_a_claim_or_focus_name(
    monkeypatch, tmp_path
):
    _install_stub_caches(monkeypatch, bars={})
    service = _focus_service(tmp_path)
    service.store.add("AMD", "short", "swing")
    service.store.add("TSLA", "long", "m5")  # an M5 focus name is NOT a swing pick
    panel = _panel(monkeypatch, tmp_path, focus_service=service)
    _claim(tmp_path, "NVDA", "LONG")
    _claim(tmp_path, "MSFT", "LONG", horizon="m5")  # not a D1 thesis

    panel._poll_pullback_watches(now=datetime.now())

    assert _pullback_watches(panel) == []


def test_the_pullback_button_never_reads_m5_bars_on_the_qt_thread(monkeypatch, tmp_path):
    panel = _panel(monkeypatch, tmp_path)
    monkeypatch.setattr(
        panel, "_m5_bars_for", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("M5 read"))
    )

    assert panel.arm_chart_watch_for("NVDA", "LONG", WATCH_KIND) is True


@pytest.mark.skip(reason="Pullback alerts are manual-only (trader 2026-09-17).")
def test_the_poll_never_arms_the_same_pick_twice(monkeypatch, tmp_path):
    _install_stub_caches(monkeypatch, bars={})
    service = _focus_service(tmp_path)
    service.store.add("AMD", "short", "swing")
    panel = _panel(monkeypatch, tmp_path, focus_service=service)
    _claim(tmp_path, "NVDA", "LONG")

    for minutes in range(4):
        panel._poll_pullback_watches(now=datetime.now() + timedelta(minutes=minutes))

    symbols = sorted(watch.symbol for watch in _pullback_watches(panel))
    assert symbols == ["AMD", "NVDA"]


@pytest.mark.skip(reason="Pullback alerts are manual-only (trader 2026-09-17).")
def test_a_watch_the_trader_disarmed_is_not_armed_again_while_the_claim_lives(
    monkeypatch, tmp_path
):
    _install_stub_caches(monkeypatch, bars={})
    service = _focus_service(tmp_path)
    service.store.add("AMD", "short", "swing")
    panel = _panel(monkeypatch, tmp_path, focus_service=service)
    _claim(tmp_path, "NVDA", "LONG")

    panel._poll_pullback_watches(now=datetime.now())
    assert sorted(w.symbol for w in _pullback_watches(panel)) == ["AMD", "NVDA"]

    panel.disarm_chart_watch_for("NVDA", WATCH_KIND)
    panel._poll_pullback_watches(now=datetime.now() + timedelta(minutes=1))
    panel._poll_pullback_watches(now=datetime.now() + timedelta(minutes=2))

    assert [w.symbol for w in _pullback_watches(panel)] == ["AMD"]


@pytest.mark.skip(reason="Pullback alerts are manual-only (trader 2026-09-17).")
def test_dropping_the_claim_retires_its_auto_watch_with_one_row(
    monkeypatch, tmp_path
):
    import claimed_picks

    _install_stub_caches(monkeypatch, bars={})
    panel = _panel(monkeypatch, tmp_path)
    _claim(tmp_path, "NVDA", "LONG")
    panel._poll_pullback_watches(now=datetime.now())
    assert [w.symbol for w in _pullback_watches(panel)] == ["NVDA"]

    claimed_picks.record_drop(
        symbol="NVDA",
        side="LONG",
        # A claim is identified by (symbol, side, SETUP) and a drop ends the
        # one it names. `_claim` above claims `pullback_sma_reclaim`, so this
        # is that claim's own retraction. (Lead ruling on the PCT-1 review,
        # 2026-09-15: the call shape here was under-specified, and giving
        # `record_drop` a blank default instead would have widened a
        # production semantic for a test's convenience.)
        claimed_setup_id="pullback_sma_reclaim",
        path=tmp_path / "claimed_picks.jsonl",
        now=datetime.now(),
    )
    panel._poll_pullback_watches(now=datetime.now() + timedelta(minutes=1))

    assert _pullback_watches(panel) == []
    rows = _events(tmp_path, "watch_retired_source_gone")
    assert len(rows) == 1
    assert rows[0]["symbol"] == "NVDA"


def test_a_hand_armed_watch_is_never_retired_by_the_auto_arm_sweep(
    monkeypatch, tmp_path
):
    """The sweep owns only what it armed. A watch the trader clicked on has no
    `auto:` source and is untouched by a claim list that never mentions it."""
    _install_stub_caches(monkeypatch, bars={})
    panel = _panel(monkeypatch, tmp_path)
    _arm_before(panel, "AAPL", "LONG", armed_at=datetime.now() - timedelta(hours=1))

    panel._poll_pullback_watches(now=datetime.now())

    assert [w.symbol for w in _pullback_watches(panel)] == ["AAPL"]
    assert _pullback_watches(panel)[0].source_text == ""


# ---------------------------------------------------------------------------
# Item 7 - the three claim names
# ---------------------------------------------------------------------------
@pytest.fixture
def pane(tmp_path):
    _qt_app()
    import pick_feedback
    from ui.widgets.alert_chart_review import AlertChartReview

    pick_feedback.clear_reviewed_today_cache()
    widget = AlertChartReview(annotations_path=tmp_path / "trader_annotations.jsonl")
    yield widget
    widget.deleteLater()


def _offered_ids(rail) -> set:
    from ui.widgets.capture_rail import _CLAIM_ROLE

    return {
        rail.setup_list.item(row).data(_CLAIM_ROLE)
        for row in range(rail.setup_list.count())
    }


def test_the_rail_offers_the_three_names_the_trader_asked_for(pane):
    """The mirror of `test_qt_alert_capture.py`'s guard, for the new ids: a
    typo in EXTRA_CLAIM_IDS would silently cost the trader a claim."""
    from ui.annotations.setup_claims import valid_setup_claim_ids
    from ui.widgets.capture_rail import EXTRA_CLAIM_IDS

    known = valid_setup_claim_ids()
    missing = [setup_id for setup_id in NEW_CLAIM_IDS if setup_id not in known]
    assert missing == [], f"the registry does not name {missing}"
    assert set(NEW_CLAIM_IDS) <= set(EXTRA_CLAIM_IDS)
    assert set(NEW_CLAIM_IDS) <= _offered_ids(pane.capture_rail)


def test_the_three_names_are_one_new_group_in_the_setup_docs():
    from setup_docs import SETUP_DOCS

    for setup_id, label in NEW_CLAIM_LABELS.items():
        doc = SETUP_DOCS[setup_id]
        assert doc["label"] == label
        assert doc["group"] == NEW_CLAIM_GROUP
        assert str(doc.get("what") or "").strip()
        assert str(doc.get("detection") or "").strip()

    # The pullback entry names the triggers it is graded on.
    detection = SETUP_DOCS["pullback_sma_reclaim"]["detection"]
    for trigger in (TRIGGER_RECLAIM, TRIGGER_THEN_LRSI, TRIGGER_RETEST):
        assert trigger in detection


def test_the_setup_registry_was_regenerated_with_the_three_names():
    import setup_registry

    for setup_id in NEW_CLAIM_IDS:
        assert setup_registry.find(setup_id) is not None, setup_id
