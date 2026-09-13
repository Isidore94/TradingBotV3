"""Packet WS-CH: more chart history without making the desk slow (WISHLIST 10H).

The trader's sentence is *"200 candles is not enough"*. The separation these
tests exist to force is between how many bars EXIST in a payload and how many
are initially VISIBLE: the daily store already holds years, and the only thing
that was short was the tail the snapshot sliced.

Every test here drives the real path - ``chart_snapshot.build_d1_snapshot``,
``ChartDataService.build_snapshots``, the real ``CandleChart`` offscreen, and a
real ``SymbolSnapshotWidget`` with a fake bot - never a helper fed a
hand-written payload.

The golden fixture ``tests/fixtures/ws_ch_chart_history_v1.json`` was pinned
from the PRE-FIX code (commit recorded inside it). Its 1,300 daily bars are
DATA, not a generator: the tests read the bars from the file so a change to the
code under test cannot move the inputs underneath the expectations.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

GOLDEN_PATH = ROOT_DIR / "tests" / "fixtures" / "ws_ch_chart_history_v1.json"

#: What the packet asks the DISPLAYED window to stay at while the payload grows.
VISIBLE_SESSIONS = 90


# --------------------------------------------------------------------------
# the pinned store
# --------------------------------------------------------------------------
def _golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _stored_bars(golden: dict, count: int | None = None) -> list[dict]:
    """The fixture's daily bars as the loader contract's dicts."""
    rows = golden["bars"]
    if count is not None:
        rows = rows[-count:]
    bars = []
    for day, open_, high, low, close, volume in rows:
        stamp = datetime.fromisoformat(day)
        bars.append(
            {
                "dt": stamp,
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }
        )
    return bars


def _history_sessions() -> int:
    """The packet's new D1 history target.

    Deliberately looked up rather than hard-coded to 1,000: the number is the
    contract's, and a test that inlined it would still pass if the constant
    never existed.
    """
    import chart_snapshot

    return int(chart_snapshot.D1_HISTORY_SESSIONS)


def _build(golden: dict, *, sessions: int, count: int | None = None) -> dict:
    import chart_snapshot

    bars = _stored_bars(golden, count)
    anchor = datetime.fromisoformat(golden["avwape_anchor"]).date()
    previous = datetime.fromisoformat(golden["avwape_prev_anchor"]).date()
    return chart_snapshot.build_d1_snapshot(
        golden["symbol"],
        sessions=sessions,
        loader=lambda _symbol: [dict(bar) for bar in bars],
        anchor_resolver=lambda _symbol: (anchor, previous),
        now=datetime.fromisoformat(golden["now"]),
    )


def _dates(bars) -> list[str]:
    return [bar["dt"].date().isoformat() for bar in bars]


# ==========================================================================
# Item 1 - the bars exist, the view is a window (the payload)
# ==========================================================================
def test_the_daily_payload_reaches_back_a_thousand_sessions_and_says_it_is_truncated():
    """1,300 sessions are stored; the chart asks for the history target.

    The store holds more than the target, so the payload is capped AND the
    snapshot says so - ``history_truncated`` is what stops the provenance strip
    claiming the oldest bar drawn is the oldest bar that exists.
    """
    golden = _golden()
    sessions = _history_sessions()
    assert sessions == 1000, "the packet's D1 target is 1,000 sessions"

    payload = _build(golden, sessions=sessions)

    assert len(payload["bars"]) == 1000
    # 1,300 stored, 1,000 shown -> the 301st stored session is the oldest drawn.
    expected_oldest = golden["bars"][-1000][0]
    assert expected_oldest == "2022-11-11", "the fixture's own 1,000th-from-last session"
    assert _dates(payload["bars"])[0] == expected_oldest
    assert payload["oldest_available"] == expected_oldest
    assert payload["history_truncated"] is True


def test_a_short_store_names_its_own_oldest_bar_and_is_not_truncated():
    """The target is a TARGET, capped by what the store holds.

    A symbol with 300 stored sessions is not "truncated": there is nothing
    further left to pan to, and saying otherwise invites the trader to drag at
    a wall.
    """
    golden = _golden()
    payload = _build(golden, sessions=_history_sessions(), count=300)

    assert len(payload["bars"]) == 300
    assert payload["oldest_available"] == golden["bars"][-300][0]
    assert payload["history_truncated"] is False


def test_the_visible_tail_indicators_are_identical_to_the_ninety_session_chart():
    """The warm-up already exists, so the last 90 values must not move.

    Every overlay is computed over the whole stored history and only then
    sliced, so lengthening the slice may not change one number in the tail the
    trader was already looking at. Golden values pinned from the pre-fix code.
    """
    golden = _golden()
    payload = _build(golden, sessions=_history_sessions())

    expected = golden["expected_90"]["overlays"]
    actual = payload["overlays"]
    assert [row["label"] for row in actual] == [row["label"] for row in expected]
    for wanted, got in zip(expected, actual):
        tail = got["values"][-VISIBLE_SESSIONS:]
        assert len(tail) == VISIBLE_SESSIONS
        assert tail == wanted["values"], f"{wanted['label']} moved in the visible tail"
    assert payload["avwape_anchor"] == golden["expected_90"]["avwape_anchor"]
    assert payload["avwape_prev_anchor"] == golden["expected_90"]["avwape_prev_anchor"]


def test_the_levels_drawn_today_survive_the_longer_payload():
    """No paint-line is lost and no price drifts when the payload grows.

    The AVWAP challenger bands are a series aligned 1:1 with the bars and the
    anchor sits inside the visible tail, so their last 90 values are the
    strongest check here: an anchor index resolved against the wrong window
    shows up as a different number, not a missing line.

    Deliberately NOT asserted: whether a level from four years ago that today's
    90-session price range filters out should now be ADMITTED. The packet says
    both "computed once over the whole payload" (item 1) and "levels unchanged"
    (its test list); that is the lead's call, and the fixture carries a far
    level (19.83) so either decision can be pinned later without new data.
    """
    import chart_levels

    golden = _golden()
    payload = _build(golden, sessions=_history_sessions())
    levels = chart_levels.build_d1_levels(
        golden["symbol"],
        payload["bars"],
        store_records=golden["store_levels"],
        trendline_feed={},
        price_alerts_path=ROOT_DIR / "tests" / "fixtures" / "no_such_price_alerts.json",
        d1_level_watches_path=ROOT_DIR / "tests" / "fixtures" / "no_such_watches.json",
        avwap_anchor=payload.get("avwape_anchor") or None,
    )
    by_id = {str(level.get("id")): level for level in levels}

    for wanted in golden["expected_90"]["levels"]:
        level_id = str(wanted["id"])
        assert level_id in by_id, f"{level_id} is no longer drawn"
        got = by_id[level_id]
        assert got["family"] == wanted["family"]
        assert got["price"] == pytest.approx(wanted["price"], abs=1e-9)
        if isinstance(wanted["values"], list):
            tail = got["values"][-VISIBLE_SESSIONS:]
            assert tail == wanted["values"], f"{level_id} moved in the visible tail"


def test_the_earnings_ribbon_marks_the_same_sessions_in_the_visible_tail():
    """An E stays on the candle that moved, and the projection does not move.

    ``indexes`` are positions, so they necessarily shift when the payload
    lengthens; the SESSIONS they name may not.
    """
    import earnings_projection

    golden = _golden()
    payload = _build(golden, sessions=_history_sessions())
    marks = earnings_projection.earnings_marks(
        golden["earnings_dates"],
        payload["bars"],
        today=datetime.fromisoformat(golden["now"]).date(),
    )
    dates = _dates(payload["bars"])
    tail = set(dates[-VISIBLE_SESSIONS:])
    marked_in_tail = sorted(
        dates[index] for index in marks["indexes"] if dates[index] in tail
    )
    assert marked_in_tail == golden["expected_90"]["earnings_marked_dates"]
    # The golden was written through json (dates as ISO strings); normalise the
    # live payload the same way rather than comparing a date to its own text.
    projected = json.loads(json.dumps(marks["projected"], default=str))
    assert projected == golden["expected_90"]["earnings_projected"]


# ==========================================================================
# Item 1 / 4 - the worker's meta and the provenance strip
# ==========================================================================
@pytest.mark.qt
def test_the_snapshot_meta_carries_the_oldest_available_date_and_the_truncation_flag():
    """The strip is fed from the worker, so the worker has to say it."""
    pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")
    from PySide6.QtCore import QThreadPool
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ui.services.chart_data_service import ChartDataService

    golden = _golden()
    bars = _stored_bars(golden)
    service = ChartDataService(store=_FakeD1Store({golden["symbol"]: bars}), pool=QThreadPool())
    try:
        _d1, _m5, meta = service.build_snapshots(golden["symbol"], [], _history_sessions())
    finally:
        service.shutdown()

    assert meta["oldest_available"] == golden["bars"][-1000][0]
    assert meta["history_truncated"] is True


def test_the_provenance_strip_names_the_oldest_date_and_that_there_is_more():
    """Item 4: in the strip the chart already has, never a popup."""
    from ui.panels.chart_review_panel import provenance_state

    base = {"source": "durable-store", "storage_tier": "durable-store"}
    truncated, _degraded = provenance_state(
        base | {"oldest_available": "2022-11-11", "history_truncated": True}
    )
    whole, _ = provenance_state(
        base | {"oldest_available": "2022-11-11", "history_truncated": False}
    )

    assert "2022-11-11" in truncated, "the trader is told how far back the chart goes"
    assert "2022-11-11" in whole
    assert truncated != whole, "'there is more behind this' has to be visible somewhere"


# ==========================================================================
# Item 1 - the chart widget: a window onto bars it already holds
# ==========================================================================
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6.QtWidgets", reason="PySide6 not installed")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def daily_chart(qapp):
    from ui.widgets.candle_chart import CandleChart

    chart = CandleChart()
    chart.resize(900, 420)
    yield chart
    chart.close()
    chart.deleteLater()
    qapp.processEvents()


def _x_range(chart) -> tuple[float, float]:
    (low, high), _y = chart.getPlotItem().vb.viewRange()
    return float(low), float(high)


def _visible_prices(chart) -> tuple[float, float]:
    _x, (low, high) = chart.getPlotItem().vb.viewRange()
    return chart.price_at(float(low)), chart.price_at(float(high))


@pytest.mark.qt
def test_the_daily_chart_opens_on_the_last_ninety_sessions_and_holds_the_rest(daily_chart):
    """A thousand candles in the widget, ninety in front of the trader.

    Panning left has to reveal the older bars with NO request, which is only
    true if the widget is already holding them - so the count and the oldest
    bar are the assertions, not a fetch counter.
    """
    golden = _golden()
    payload = _build(golden, sessions=1000)

    daily_chart.set_data(
        payload["bars"],
        payload["overlays"],
        timeframe="d1",
        initial_view_sessions=VISIBLE_SESSIONS,
    )

    assert daily_chart.bar_count() == 1000
    assert daily_chart.bar_at(0)["dt"].date().isoformat() == "2022-11-11"
    low, high = _x_range(daily_chart)
    assert high >= 998, "the newest candle is on screen"
    assert round(high - low) == pytest.approx(VISIBLE_SESSIONS, abs=6), (
        f"the opening window is {round(high - low)} bars wide, not ~90"
    )
    assert low > 500, "the opening view is the tail, not four years squeezed flat"


@pytest.mark.qt
def test_the_opening_price_scale_comes_from_the_visible_window(daily_chart):
    """A 4-year y-range would flatten today's candles into a line.

    The fixture walks from ~20 to ~197 on purpose: the last 90 sessions live
    between 163.89 and 196.74 while the whole 1,000-bar payload starts at
    39.30, so a scale taken from the wrong window is a number, not a rounding.
    """
    golden = _golden()
    payload = _build(golden, sessions=1000)
    tail = payload["bars"][-VISIBLE_SESSIONS:]
    tail_low = min(bar["low"] for bar in tail)
    tail_high = max(bar["high"] for bar in tail)

    daily_chart.set_data(
        payload["bars"],
        payload["overlays"],
        timeframe="d1",
        initial_view_sessions=VISIBLE_SESSIONS,
    )

    low, high = _visible_prices(daily_chart)
    assert low > 100.0, f"the y-range reaches down to {low:.2f} - that is the 2022 price"
    assert low == pytest.approx(tail_low, rel=0.20)
    assert high == pytest.approx(tail_high, rel=0.20)


@pytest.mark.qt
def test_an_intraday_chart_without_a_history_window_still_shows_every_bar(daily_chart):
    """Characterization: M5 keeps today's whole-frame behaviour."""
    golden = _golden()
    bars = _stored_bars(golden, 120)

    daily_chart.set_data(bars, [], timeframe="m5")

    low, high = _x_range(daily_chart)
    assert low < 0.5
    assert high >= 119


@pytest.mark.qt
def test_a_thousand_candles_paint_inside_the_frame_budget(daily_chart, capsys):
    """The packet asks for the paint cost at 1,000 bars to be measured.

    The ceiling is deliberately loose - this is a regression tripwire, not a
    benchmark. The measured number is printed for the packet's record.
    """
    from PySide6.QtGui import QPixmap

    golden = _golden()
    payload = _build(golden, sessions=1000)

    started = time.perf_counter()
    daily_chart.set_data(payload["bars"], payload["overlays"], timeframe="d1")
    set_data_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    pixmap: QPixmap = daily_chart.grab()
    paint_ms = (time.perf_counter() - started) * 1000.0

    with capsys.disabled():
        print(
            f"\n[WS-CH] 1,000 D1 candles + {len(payload['overlays'])} overlays: "
            f"set_data {set_data_ms:.1f} ms, paint {paint_ms:.1f} ms"
        )
    assert not pixmap.isNull()
    assert set_data_ms < 1500.0
    assert paint_ms < 1500.0


# ==========================================================================
# Item 2 - M5 in bounded chunks
# ==========================================================================
M5_SESSION_DATES = [
    "2026-09-01",
    "2026-09-02",
    "2026-09-03",
    "2026-09-04",
    "2026-09-08",
    "2026-09-09",
    "2026-09-10",
    "2026-09-11",
    "2026-09-14",
    "2026-09-15",
    "2026-09-16",
    "2026-09-17",
    "2026-09-18",
    "2026-09-21",
]
#: A full RTH session of five-minute candles, 09:30 through 15:55.
BARS_PER_SESSION = 78


class FakeBot:
    """BounceBot's chart-bar seam, and nothing else.

    ``m5_chart_bars`` is documented as an in-memory read of the scan loop's
    cache that never fetches; this records every call so a test can prove the
    paint path made none, and can be made to raise so a test can prove a
    provider failure costs the older bars and not the chart.
    """

    def __init__(self, sessions_available: int = 14, base: float = 100.0) -> None:
        self.sessions_available = int(sessions_available)
        self.base = float(base)
        self.calls: list[tuple[str, int]] = []
        self.fail_beyond: int | None = None
        self.latest_bars: dict = {}

    def bars_for(self, symbol: str, sessions: int) -> list[dict]:
        wanted = M5_SESSION_DATES[: min(int(sessions), self.sessions_available)]
        offset = sum(ord(char) for char in str(symbol).upper()) % 37
        bars: list[dict] = []
        for day_index, day in enumerate(reversed(wanted)):
            start = datetime.fromisoformat(day) + timedelta(hours=9, minutes=30)
            price = self.base + offset + day_index
            for index in range(BARS_PER_SESSION):
                stamp = start + timedelta(minutes=5 * index)
                close = price + (index % 7) * 0.05
                bars.append(
                    {
                        "dt": stamp,
                        "open": round(close - 0.02, 4),
                        "high": round(close + 0.06, 4),
                        "low": round(close - 0.08, 4),
                        "close": round(close, 4),
                        "volume": 10_000.0 + index,
                    }
                )
        bars.sort(key=lambda bar: bar["dt"])
        return bars

    def m5_chart_bars(self, symbol, max_sessions=2):
        sessions = int(max_sessions)
        self.calls.append((str(symbol).upper(), sessions))
        if self.fail_beyond is not None and sessions > self.fail_beyond:
            raise RuntimeError("provider down")
        return self.bars_for(symbol, sessions)


class _FakeSeries:
    """What ``D1BarStore.load`` hands the snapshot loader."""

    def __init__(self, bars: list[dict], source: str = "shared") -> None:
        self._bars = bars
        self.source = source

    def __len__(self) -> int:
        return len(self._bars)

    def as_bar_dicts(self) -> list[dict]:
        return [dict(bar) for bar in self._bars]


class _FakeD1Store:
    def __init__(self, by_symbol: dict[str, list[dict]]) -> None:
        self._by_symbol = {key.upper(): value for key, value in by_symbol.items()}

    def load(self, symbol: str):
        bars = self._by_symbol.get(str(symbol).upper())
        return _FakeSeries(bars) if bars else None

    def cached(self, symbol: str):
        return self.load(symbol)

    def prefetch(self, symbols) -> int:
        return 0


@pytest.fixture
def snapshot_widget(qapp, monkeypatch):
    """A real SymbolSnapshotWidget with every provider door closed.

    The centre Visual Alert Review chart builds exactly this widget
    (``AlertChartReview.__init__``: ``SymbolSnapshotWidget(self, compact=True)``
    with no ``d1_sessions``), so the default configuration is the one under
    test.
    """
    from ui.services import chart_bar_refresh
    from ui.services import chart_data_service as service_mod
    from ui.widgets import symbol_snapshot_dialog as mod

    golden = _golden()
    store = _FakeD1Store({golden["symbol"]: _stored_bars(golden)})
    monkeypatch.setattr(service_mod, "shared_store", lambda: store)

    class _NoRefresh:
        def best_bars(self, _symbol, bars):
            return bars

    monkeypatch.setattr(chart_bar_refresh, "shared_refresh_service", lambda: _NoRefresh())
    # Both background repairs start a THREAD that reaches a provider. Neither
    # is what this packet is about, and a test may not put the suite on the
    # network.
    monkeypatch.setattr(mod.SymbolSnapshotWidget, "_start_d1_backfill", lambda self, symbol: None)
    monkeypatch.setattr(mod.SymbolSnapshotWidget, "_start_forming_fetch", lambda self, symbol: None)

    widget = mod.SymbolSnapshotWidget(compact=True)
    widget.resize(900, 700)
    yield widget
    widget._data.shutdown()
    widget.deleteLater()
    qapp.processEvents()


def _drain(widget, app, rounds: int = 8) -> None:
    for _ in range(rounds):
        widget._data.wait_for_idle(5000)
        app.processEvents()


def _older_button(widget):
    """The M5 chart's 'Load older' control, whatever it currently says.

    Matched on the word the trader reads rather than an attribute name: after
    a provider failure the same button reads 'older bars unavailable'.
    """
    from PySide6.QtWidgets import QPushButton

    found = [
        button
        for button in widget.findChildren(QPushButton)
        if "older" in (button.text() or "").strip().lower()
    ]
    return found[0] if found else None


def _require_older_button(widget):
    button = _older_button(widget)
    if button is None:
        pytest.fail(
            "no 'Load older' button on the M5 chart "
            "(SymbolSnapshotWidget hosts the M5 chart for both the centre "
            "review pane and Chart Review)"
        )
    return button


def _m5_session_count(widget) -> int:
    return len({bar["dt"].date() for bar in widget.cached_m5_bars()})


@pytest.mark.qt
def test_the_centre_review_chart_holds_a_thousand_daily_candles(snapshot_widget, qapp):
    """Item 1 end to end: the default widget, the store, the drawn chart."""
    golden = _golden()

    snapshot_widget.set_symbol(golden["symbol"])
    _drain(snapshot_widget, qapp)

    assert snapshot_widget.d1_chart.bar_count() == 1000
    assert snapshot_widget.d1_chart.bar_at(0)["dt"].date().isoformat() == "2022-11-11"
    low, high = _x_range(snapshot_widget.d1_chart)
    assert round(high - low) == pytest.approx(VISIBLE_SESSIONS, abs=6), (
        "the payload grew but the opening view did not stay at 90 sessions"
    )


@pytest.mark.qt
def test_load_older_extends_the_intraday_chart_by_two_sessions_without_duplicating_a_bar(
    snapshot_widget, qapp
):
    bot = FakeBot()
    snapshot_widget.set_symbol("AAA", bot=bot)
    _drain(snapshot_widget, qapp)
    assert _m5_session_count(snapshot_widget) == 2, "today's default is two sessions"

    _require_older_button(snapshot_widget).click()
    _drain(snapshot_widget, qapp)

    bars = snapshot_widget.cached_m5_bars()
    assert _m5_session_count(snapshot_widget) == 4
    assert len(bars) == 4 * BARS_PER_SESSION
    stamps = [bar["dt"] for bar in bars]
    assert len(set(stamps)) == len(stamps), "an overlapping chunk was merged twice"
    assert stamps == sorted(stamps)


@pytest.mark.qt
def test_load_older_leaves_the_visible_candles_where_they_were(snapshot_widget, qapp):
    """The older bars arrive on the LEFT, so the index range has to move.

    Restoring the x-range verbatim would slide the trader 78 candles back
    through their own chart; what must be preserved is the CANDLES they were
    looking at, which is what this asserts.
    """
    bot = FakeBot()
    snapshot_widget.set_symbol("AAA", bot=bot)
    _drain(snapshot_widget, qapp)

    chart = snapshot_widget.m5_chart
    chart.getPlotItem().setXRange(100.0, 140.0, padding=0)
    qapp.processEvents()

    def visible_edges():
        low, high = _x_range(chart)
        first = max(0, int(math.ceil(low)))
        last = min(chart.bar_count() - 1, int(math.floor(high)))
        return chart.bar_at(first)["dt"], chart.bar_at(last)["dt"]

    before = visible_edges()

    _require_older_button(snapshot_widget).click()
    _drain(snapshot_widget, qapp)

    assert chart.bar_count() == 4 * BARS_PER_SESSION
    assert visible_edges() == before


@pytest.mark.qt
def test_a_load_older_result_for_a_symbol_the_trader_left_is_dropped(snapshot_widget, qapp):
    """A stale token, not a repaint: the chart shows what is in front of them."""
    bot = FakeBot()
    snapshot_widget.set_symbol("AAA", bot=bot)
    _drain(snapshot_widget, qapp)

    _require_older_button(snapshot_widget).click()
    snapshot_widget.set_symbol("BBB", bot=bot)
    _drain(snapshot_widget, qapp)

    drawn = snapshot_widget.cached_m5_bars()
    assert drawn, "BBB should be drawn"
    expected = {bar["close"] for bar in bot.bars_for("BBB", 2)}
    assert {bar["close"] for bar in drawn} <= expected, "AAA's older bars landed on BBB's chart"
    assert _m5_session_count(snapshot_widget) == 2, "the new symbol starts at two sessions"


@pytest.mark.qt
def test_a_failing_provider_keeps_the_bars_and_says_older_bars_unavailable(
    snapshot_widget, qapp
):
    """A raise must cost the older bars, never the chart the trader has."""
    bot = FakeBot()
    snapshot_widget.set_symbol("AAA", bot=bot)
    _drain(snapshot_widget, qapp)
    before = list(snapshot_widget.cached_m5_bars())
    assert before

    bot.fail_beyond = 2
    _require_older_button(snapshot_widget).click()
    _drain(snapshot_widget, qapp)

    assert snapshot_widget.cached_m5_bars() == before, "the chart was emptied by a failure"
    assert snapshot_widget.m5_chart.bar_count() == len(before)
    assert "older bars unavailable" in (_require_older_button(snapshot_widget).text() or "").lower()


@pytest.mark.qt
def test_load_older_stops_at_ten_sessions_for_one_symbol(snapshot_widget, qapp):
    """The cap is what keeps the cache from ballooning over a desk session."""
    bot = FakeBot(sessions_available=14)
    snapshot_widget.set_symbol("AAA", bot=bot)
    _drain(snapshot_widget, qapp)

    for _click in range(6):
        button = _older_button(snapshot_widget)
        if button is None or not button.isEnabled():
            break
        button.click()
        _drain(snapshot_widget, qapp)

    assert _m5_session_count(snapshot_widget) == 10
    assert max(sessions for _symbol, sessions in bot.calls) <= 10, (
        "the bot was asked for more sessions than the cap allows"
    )


@pytest.mark.qt
def test_repainting_the_intraday_chart_never_reads_the_bar_cache(snapshot_widget, qapp):
    """Characterization, and the guard on the pan-left trigger.

    A pan that fetches is a fetch on the paint path. The bot is armed to raise
    on ANY call, so a read from inside a repaint is a failure rather than a
    slow frame.
    """
    bot = FakeBot()
    snapshot_widget.set_symbol("AAA", bot=bot)
    _drain(snapshot_widget, qapp)
    drawn = len(snapshot_widget.cached_m5_bars())
    assert drawn

    calls_before = len(bot.calls)
    bot.fail_beyond = 0  # any call at all now raises
    chart = snapshot_widget.m5_chart
    chart.resize(640, 320)
    chart.getPlotItem().setXRange(0.0, 40.0, padding=0)
    pixmap = chart.grab()
    qapp.processEvents()

    assert not pixmap.isNull()
    assert len(bot.calls) == calls_before, "the paint path read the bar cache"
    assert len(snapshot_widget.cached_m5_bars()) == drawn


# ==========================================================================
# The trader's "never raise every provider request blindly"
# ==========================================================================
@pytest.mark.qt
def test_one_chart_click_never_asks_the_provider_for_four_years_of_daily_bars(
    qapp, monkeypatch
):
    """The stale-store repair is a repair, not a history import.

    Today the widget asks for ``max(260, ceil(sessions * 365 / 252))`` calendar
    days - 260 for the default view, 754 for Chart Review's 520 sessions. Wiring
    a 1,000-session history target straight into that field would make every
    click on a stale symbol a 1,449-day Yahoo request. The store is filled by
    the scan pipeline; this path only catches a symbol up.
    """
    from ui.services import safe_import
    from ui.widgets import symbol_snapshot_dialog as mod

    requested: list[int] = []

    class _Legacy:
        @staticmethod
        def fetch_daily_bars(_client, _symbol, days):
            requested.append(int(days))

    monkeypatch.setattr(safe_import, "master_avwap_legacy", lambda: _Legacy())
    mod._D1_BACKFILL_ATTEMPTS.clear()

    widget = mod.SymbolSnapshotWidget(compact=True)
    try:
        widget._symbol = "WSCH"
        widget._start_d1_backfill("WSCH")
        thread = widget._d1_backfill_thread
        assert thread is not None
        thread.join(10)
        qapp.processEvents()
    finally:
        widget._data.shutdown()
        widget.deleteLater()
        qapp.processEvents()

    assert requested, "the backfill never reached the provider seam"
    assert max(requested) <= 800, (
        f"one chart click asked the provider for {max(requested)} calendar days"
    )
