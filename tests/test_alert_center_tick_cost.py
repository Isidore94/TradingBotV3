"""Desk snappiness packet 2, item 1: the minute tick stops redoing itself.

The 2026-08-31 stall log measured 8,008 GUI freezes / ~78 minutes in one day.
Packet 1 took the three largest causes; this file pins the next three, all of
them the same shape - identical work repeated per symbol, per kind, or per
watch on a 30 s / 60 s timer over a ~105-symbol Focus set.

What these defend is that the repetition is gone and **nothing else moved**:
the bars are still `m5_chart_bars`'s own output, the levels are still
`d1_event_levels`' own output, and the prefetch still asks for exactly the same
symbols. No detector, gate, threshold or alert rule is touched by any of it.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.qt

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _Bar:
    __slots__ = ("dt", "open", "high", "low", "close", "volume")

    def __init__(self, dt, price=100.0):
        self.dt = dt
        self.open = price
        self.high = price + 1
        self.low = price - 1
        self.close = price
        self.volume = 1000


class _Bot:
    """Just enough BounceBot to exercise the memo: the real key lookup and the
    real materialization, counted."""

    def __init__(self, symbol="NVDA", count=150):
        start = datetime(2026, 8, 31, 6, 30)
        self.latest_bars = {
            f"{symbol}|5 D|5 mins": [_Bar(start + timedelta(minutes=5 * i)) for i in range(count)]
        }
        self.calls = 0

    def m5_chart_bars(self, symbol, max_sessions=2):
        self.calls += 1
        bars = self.latest_bars.get(f"{str(symbol).upper()}|5 D|5 mins") or []
        return [
            {
                "dt": bar.dt,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            for bar in bars
        ]


class _BounceService:
    def __init__(self, bot):
        self._bot = bot

    def current_bot(self):
        return self._bot


def _panel(tmp_path):
    from ui.panels.alert_center_panel import AlertCenterPanel

    return AlertCenterPanel(
        parked_symbols_path=tmp_path / "parked.json",
        focus_d1_flags_path=tmp_path / "focus_flags.json",
    )


class TestTheM5BarsAreMaterializedOnce:
    def test_a_second_call_over_unchanged_bars_materializes_nothing(self, tmp_path):
        """Eight timer-driven sites ask for the same symbol's bars per tick and
        each call rebuilt ~150 dicts with six float() coercions apiece."""
        panel = _panel(tmp_path)
        bot = _Bot()
        panel._bounce_service = _BounceService(bot)

        first = panel._m5_bars_for("NVDA")
        second = panel._m5_bars_for("NVDA")

        assert bot.calls == 1
        assert second is first, "the same list, not an equal copy"
        assert len(first) == 150

    def test_a_new_bar_invalidates_the_memo(self, tmp_path):
        """Appended in place: the list object is the same, so length and the
        last stamp are what has to catch it."""
        panel = _panel(tmp_path)
        bot = _Bot()
        panel._bounce_service = _BounceService(bot)

        before = panel._m5_bars_for("NVDA")
        source = bot.latest_bars["NVDA|5 D|5 mins"]
        source.append(_Bar(source[-1].dt + timedelta(minutes=5)))
        after = panel._m5_bars_for("NVDA")

        assert bot.calls == 2
        assert len(after) == len(before) + 1

    def test_a_replaced_series_invalidates_the_memo(self, tmp_path):
        """Refreshed symbols arrive as a NEW list; identity catches those."""
        panel = _panel(tmp_path)
        bot = _Bot()
        panel._bounce_service = _BounceService(bot)

        panel._m5_bars_for("NVDA")
        bot.latest_bars["NVDA|5 D|5 mins"] = list(bot.latest_bars["NVDA|5 D|5 mins"])
        panel._m5_bars_for("NVDA")

        assert bot.calls == 2

    def test_the_two_session_counts_are_cached_apart(self, tmp_path):
        """`sessions=2` is a different question (an ATR(14) needs warm-up bars)
        and must not be answered from the one-session entry."""
        panel = _panel(tmp_path)
        bot = _Bot()
        panel._bounce_service = _BounceService(bot)

        panel._m5_bars_for("NVDA", sessions=1)
        panel._m5_bars_for("NVDA", sessions=2)
        panel._m5_bars_for("NVDA", sessions=1)
        panel._m5_bars_for("NVDA", sessions=2)

        assert bot.calls == 2

    def test_the_memo_is_bounded(self, tmp_path):
        from ui.panels import alert_center_panel as panel_mod

        panel = _panel(tmp_path)
        bot = _Bot()
        panel._bounce_service = _BounceService(bot)
        for index in range(panel_mod.M5_BAR_DICT_CACHE_LIMIT + 20):
            panel._m5_bars_for(f"SYM{index}")

        assert len(panel._m5_bar_dicts) <= panel_mod.M5_BAR_DICT_CACHE_LIMIT

    def test_no_bot_still_answers_empty(self, tmp_path):
        panel = _panel(tmp_path)
        panel._bounce_service = None
        assert panel._m5_bars_for("NVDA") == []


class TestTheD1LevelsAreBuiltOncePerSymbol:
    @staticmethod
    def _d1_bars(count=260):
        start = date(2025, 8, 1)
        bars = []
        for index in range(count):
            price = 100.0 + index * 0.1
            bars.append(
                {
                    "dt": datetime.combine(start + timedelta(days=index), datetime.min.time()),
                    "open": price,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 1_000_000,
                }
            )
        return bars

    def test_a_shared_cache_builds_the_levels_once_for_ten_kinds(self, monkeypatch):
        """`d1_event_levels` sorts ~490 bars and builds 5d/20d extremes, three
        SMAs, an EMA15 recursion and the AVWAP bands. Ten kinds per symbol used
        to re-enter it ten times with identical arguments."""
        import chart_watch

        real = chart_watch.d1_event_levels
        calls = []

        def counted(d1_bars, *, session, avwape_anchor=None):
            calls.append((session, avwape_anchor))
            return real(d1_bars, session=session, avwape_anchor=avwape_anchor)

        monkeypatch.setattr(chart_watch, "d1_event_levels", counted)

        d1_bars = self._d1_bars()
        moment = datetime(2026, 4, 20, 10, 0)
        m5_bars = [
            {"dt": moment - timedelta(minutes=20), "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5},
            {"dt": moment - timedelta(minutes=15), "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0},
        ]
        cache: dict = {}
        for kind in sorted(chart_watch.D1_EVENT_KINDS):
            watch = chart_watch.D1EventWatch(
                symbol="NVDA", kind=kind, armed_at=moment - timedelta(hours=2)
            )
            chart_watch.evaluate_d1_event_watch(
                watch, m5_bars, d1_bars, now=moment, levels_cache=cache
            )

        assert len(chart_watch.D1_EVENT_KINDS) >= 5
        assert len(calls) == 1, f"one build for {len(chart_watch.D1_EVENT_KINDS)} kinds, got {calls}"

    def test_the_cached_result_is_identical_to_the_per_kind_path(self):
        """Behaviour-identical by construction: with no cache it is the call it
        replaced, and with one it is the same object."""
        import chart_watch

        d1_bars = self._d1_bars()
        moment = datetime(2026, 4, 20, 10, 0)
        m5_bars = [
            {"dt": moment - timedelta(minutes=20), "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5},
        ]
        watch = chart_watch.D1EventWatch(
            symbol="NVDA", kind="new_20d_high", armed_at=moment - timedelta(hours=2)
        )

        uncached = chart_watch.evaluate_d1_event_watch(watch, m5_bars, d1_bars, now=moment)
        cached = chart_watch.evaluate_d1_event_watch(
            watch, m5_bars, d1_bars, now=moment, levels_cache={}
        )

        assert uncached == cached

    def test_the_anchor_is_part_of_the_key(self):
        """The AVWAPE kinds are given an anchor and the others are not, so one
        cache must be able to hold both without either seeing the other's."""
        import chart_watch

        cache: dict = {}
        d1_bars = self._d1_bars()
        session = date(2026, 4, 20)
        anchored = chart_watch._cached_d1_event_levels(cache, d1_bars, session, date(2026, 2, 1))
        plain = chart_watch._cached_d1_event_levels(cache, d1_bars, session, None)

        assert len(cache) == 2
        assert anchored is not plain
        assert chart_watch._cached_d1_event_levels(cache, d1_bars, session, None) is plain


class TestThePrefetchIsBatched:
    def test_many_symbols_become_one_prefetch(self, tmp_path, monkeypatch):
        """~105 single-element tasks per minute queued ahead of the snapshot
        for the chart the trader had just clicked, in a 2-thread pool."""
        from ui.services import chart_data_service

        requests = []

        class _Service:
            def cached_series(self, symbol):
                return None

            def cached_bar_dicts(self, symbol):
                return []

            def prefetch(self, symbols):
                requests.append(list(symbols))

        monkeypatch.setattr(chart_data_service, "shared_service", lambda: _Service())
        panel = _panel(tmp_path)

        for symbol in ("NVDA", "AMD", "TSLA", "NVDA"):
            panel._d1_bars_for(symbol)

        assert requests == [], "queued, not issued one at a time"
        panel._flush_d1_prefetch()
        assert requests == [["NVDA", "AMD", "TSLA"]], "one task, each symbol once"

    def test_the_flush_is_idempotent(self, tmp_path, monkeypatch):
        from ui.services import chart_data_service

        requests = []

        class _Service:
            def cached_series(self, symbol):
                return None

            def cached_bar_dicts(self, symbol):
                return []

            def prefetch(self, symbols):
                requests.append(list(symbols))

        monkeypatch.setattr(chart_data_service, "shared_service", lambda: _Service())
        panel = _panel(tmp_path)
        panel._d1_bars_for("NVDA")
        panel._flush_d1_prefetch()
        panel._flush_d1_prefetch()

        assert requests == [["NVDA"]]


class TestTheAnyBounceWatchAsksOnce:
    def test_one_bar_read_per_watch_not_two(self, tmp_path, monkeypatch):
        """The levels builder and the evaluation both need today's M5 bars."""
        from chart_watch import ANY_BOUNCE_KINDS, AnyBounceWatch

        from ui.panels import alert_center_panel as panel_mod

        panel = _panel(tmp_path)
        asked = []

        monkeypatch.setattr(
            panel, "_m5_bars_for", lambda symbol, **kwargs: asked.append(symbol) or []
        )
        monkeypatch.setattr(panel, "_d1_bars_for", lambda symbol: [])
        # Levels have to come back truthy or the evaluation - the second read -
        # is skipped and the test proves nothing.
        monkeypatch.setattr(panel_mod, "any_bounce_levels", lambda **kwargs: {"vwap": 100.0})
        monkeypatch.setattr(panel_mod, "evaluate_any_bounce_watch", lambda *a, **k: None)
        panel._any_bounce_watches = [
            AnyBounceWatch(
                symbol="NVDA",
                side="long",
                kinds=tuple(ANY_BOUNCE_KINDS),
                armed_at=datetime.now(),
            )
        ]

        panel._poll_any_bounce_watches(now=datetime.now())

        assert asked == ["NVDA"]
