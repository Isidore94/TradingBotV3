"""R5 sections 3.2 and 3.3 wiring: the confluence and first-candle ORB engines.

The engine-level rules live in ``test_m5_signal_engines.py``. What is asserted
here is what only the detector can be asked:

**They are off.** The spec's section 8.2 held these two engines out of the live
loop until a desk session measured section 3.1's volume. The trader overrode
that ordering on 2026-08-18 to integrate the packet; what replaces it is that
every new type defaults OFF, so the desk session now gates audibility rather
than existence. A test says so, because a default is exactly the kind of thing
a later edit flips without noticing.

**The confluence is Focus-scoped.** The trader asked for the "strongest" tell
on names they are already watching. A watchlist name that is not in M5 Focus
must produce nothing, however perfect its chart.

**The ORB candidate is an annotation, not a trade.** It never seeds the bounce
outcome tracker; only the re-break does. Measuring an engine against events it
never claimed were entries would corrupt the evidence the promotion ladder
reads.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from bounce_bot_lib import legacy  # noqa: E402
from m5_signal_engines import M5_SIGNAL_TAG  # noqa: E402

OPEN = datetime(2026, 8, 17, 6, 30)
PRIOR_SESSION = OPEN - timedelta(days=3)


@dataclass
class StubBar:
    dt: datetime
    open: float
    high: float
    low: float
    close: float


def ohlc_bars(rows, *, start=OPEN):
    return [
        StubBar(start + timedelta(minutes=5 * index), row[0], row[1], row[2], row[3])
        for index, row in enumerate(rows)
    ]


def decline_then_reversal():
    """The confluence series from the engine tests: legs at 22, 23 and 26."""
    rows = []
    price = 120.0
    for _ in range(22):
        close = price - 0.8
        rows.append((price, price + 0.15, close - 0.15, close))
        price = close
    for _ in range(8):
        close = price + 1.2
        rows.append((price, close + 0.2, price - 0.1, close))
        price = close
    return rows


def orb_series():
    """The ORB series from the engine tests: candidate 10, recross 22, break 24."""
    prior = [(100.0, 100.2, 99.8, 100.0)] * 10
    session = [(103.0, 104.5, 102.9, 104.2)]
    price = 104.2
    for _ in range(8):
        close = price - 0.5
        session.append((price, price + 0.1, close - 0.1, close))
        price = close
    for _ in range(7):
        close = price + 0.75
        session.append((price, close + 0.15, price - 0.1, close))
        price = close
    return ohlc_bars(prior, start=PRIOR_SESSION) + ohlc_bars(session, start=OPEN)


def stub_bot(symbol_bars, *, longs=(), shorts=(), focus_long=(), focus_short=()):
    """A BounceBot with every heavy collaborator replaced by a stub."""
    bot = legacy.BounceBot.__new__(legacy.BounceBot)
    bot._lrsi_cross_state = None
    bot._confluence_state = None
    bot._orb_first_candle_state = None
    bot.m5_signal_toggles = dict(legacy.M5_SIGNAL_TYPE_DEFAULTS)
    bot.market_environment_lock = threading.Lock()
    bot.emitted = []
    bot.logged = []
    bot.registered = []

    spy = ohlc_bars([(100.0, 100.0, 100.0, 100.0)] * 3)
    bot._spy_session_bars = lambda: (spy, None)
    bot._watchlist_day_sweep_symbols = lambda side, symbols=None: (
        list(longs) if side == "long" else list(shorts)
    )
    bot._human_focus_sets = lambda: {
        "long": set(focus_long),
        "short": set(focus_short),
    }
    bot.get_cached_5m_bars = lambda symbol: symbol_bars.get(symbol, [])
    bot._log_bounce_candidate_event = lambda *a, **k: {"event_id": "evt-1"}
    bot._register_bounce_outcome = lambda *a, **k: bot.registered.append(a[:2])
    bot._evaluate_bounce_alert_quality = lambda *a, **k: {"tier": "B"}
    bot._measured_exit_suffix = lambda *a, **k: ""
    bot.log_symbol = lambda symbol, text: bot.logged.append((symbol, text))
    bot.log_bounce_to_file = lambda **k: None
    bot.gui_callback = lambda payload, tag: bot.emitted.append((payload, tag))
    return bot


def enable(bot, *types):
    for signal_type in types:
        bot.set_m5_signal_enabled(signal_type, True)


@pytest.fixture
def at_the_confluence(monkeypatch):
    """Bar 26 -- the LRSI leg -- is the last completed bar."""
    stamp = OPEN + timedelta(minutes=5 * 27)
    monkeypatch.setattr(legacy, "get_market_local_now", lambda *a, **k: stamp)
    return stamp


@pytest.fixture
def at_the_orb_break(monkeypatch):
    """Session bar 14 (the re-break) is the last completed bar."""
    stamp = OPEN + timedelta(minutes=5 * 15)
    monkeypatch.setattr(legacy, "get_market_local_now", lambda *a, **k: stamp)
    return stamp


class TestTheyShipOff:
    def test_every_new_engine_defaults_off(self):
        for signal_type in (
            legacy.M5_CONFLUENCE_TYPE,
            legacy.ORB_CANDIDATE_TYPE,
            legacy.ORB_NEW_EXTREME_TYPE,
            legacy.ORB_LRSI_RECROSS_TYPE,
        ):
            assert legacy.M5_SIGNAL_TYPE_DEFAULTS[signal_type] is False

    def test_the_proven_engine_is_untouched(self):
        """Turning two engines on probation must not disturb the shipped one."""
        assert legacy.M5_SIGNAL_TYPE_DEFAULTS[legacy.LRSI_CROSS_20_TYPE] is True
        assert legacy.M5_SIGNAL_TYPE_DEFAULTS[legacy.LRSI_CROSS_50_TYPE] is True

    def test_they_are_not_learning_types(self):
        for signal_type in legacy.M5_SIGNAL_TYPE_DEFAULTS:
            assert signal_type not in legacy.BOUNCE_TYPE_DEFAULTS
            assert signal_type not in legacy.BOUNCE_LEARNING_TYPE_KEYS

    def test_an_off_confluence_sweep_does_no_work(self, at_the_confluence):
        bars = ohlc_bars(decline_then_reversal())
        bot = stub_bot({"AAA": bars}, longs=["AAA"], focus_long=["AAA"])
        assert bot.check_confluence_setups() == []
        assert bot.emitted == []

    def test_an_off_orb_sweep_does_no_work(self, at_the_orb_break):
        bot = stub_bot({"AAA": orb_series()}, longs=["AAA"])
        assert bot.check_orb_first_candle_setups() == []
        assert bot.emitted == []


class TestTheConfluenceSweep:
    def test_a_focus_name_alerts_once(self, at_the_confluence):
        bars = ohlc_bars(decline_then_reversal())
        bot = stub_bot({"AAA": bars}, longs=["AAA"], focus_long=["AAA"])
        enable(bot, legacy.M5_CONFLUENCE_TYPE)

        assert len(bot.check_confluence_setups()) == 1
        assert len(bot.emitted) == 1
        payload, tag = bot.emitted[0]
        assert tag == M5_SIGNAL_TAG
        assert "M5 CONFLUENCE AAA" in payload["text"]

        # Same bar, next scan cycle.
        assert bot.check_confluence_setups() == []
        assert len(bot.emitted) == 1

    def test_a_watchlist_name_outside_focus_is_silence(self, at_the_confluence):
        """The trader asked for this one on names they are already watching."""
        bars = ohlc_bars(decline_then_reversal())
        bot = stub_bot({"AAA": bars}, longs=["AAA"], focus_long=[])
        enable(bot, legacy.M5_CONFLUENCE_TYPE)

        assert bot.check_confluence_setups() == []

    def test_a_forming_bar_is_not_an_alert(self, monkeypatch):
        forming = OPEN + timedelta(minutes=5 * 26 + 4)
        monkeypatch.setattr(legacy, "get_market_local_now", lambda *a, **k: forming)
        bars = ohlc_bars(decline_then_reversal())
        bot = stub_bot({"AAA": bars}, longs=["AAA"], focus_long=["AAA"])
        enable(bot, legacy.M5_CONFLUENCE_TYPE)

        assert bot.check_confluence_setups() == []

    def test_the_short_side_mirrors(self, at_the_confluence):
        mirrored = [
            (200.0 - row[0], 200.0 - row[2], 200.0 - row[1], 200.0 - row[3])
            for row in decline_then_reversal()
        ]
        bot = stub_bot(
            {"ZZZ": ohlc_bars(mirrored)}, shorts=["ZZZ"], focus_short=["ZZZ"]
        )
        enable(bot, legacy.M5_CONFLUENCE_TYPE)

        assert len(bot.check_confluence_setups()) == 1
        payload, _tag = bot.emitted[0]
        assert "(short)" in payload["text"]

    def test_the_state_resets_on_a_new_session(self, at_the_confluence):
        bars = ohlc_bars(decline_then_reversal())
        bot = stub_bot({"AAA": bars}, longs=["AAA"], focus_long=["AAA"])
        enable(bot, legacy.M5_CONFLUENCE_TYPE)
        bot.check_confluence_setups()
        assert len(bot.emitted) == 1

        bot._confluence_state["date"] = OPEN.date() - timedelta(days=1)
        bot.check_confluence_setups()
        assert len(bot.emitted) == 2

    def test_no_spy_session_means_no_sweep(self, at_the_confluence):
        bars = ohlc_bars(decline_then_reversal())
        bot = stub_bot({"AAA": bars}, longs=["AAA"], focus_long=["AAA"])
        enable(bot, legacy.M5_CONFLUENCE_TYPE)
        bot._spy_session_bars = lambda: ([], None)

        assert bot.check_confluence_setups() == []


class TestTheOrbSweep:
    def test_the_break_alerts_and_is_measurable(self, at_the_orb_break):
        bot = stub_bot({"AAA": orb_series()}, longs=["AAA"])
        enable(bot, legacy.ORB_NEW_EXTREME_TYPE)

        events = bot.check_orb_first_candle_setups()
        assert [event.kind for event in events] == ["new_extreme"]
        payload, tag = bot.emitted[0]
        assert tag == M5_SIGNAL_TAG
        assert "ORB NEW HIGH AAA" in payload["text"]
        # The break asserts a move, so it seeds the outcome tracker.
        assert bot.registered

    def test_the_candidate_mark_is_an_annotation_not_an_outcome(self, monkeypatch):
        """It says "this name opened the way the setup wants" - nothing more."""
        stamp = OPEN + timedelta(minutes=5)
        monkeypatch.setattr(legacy, "get_market_local_now", lambda *a, **k: stamp)
        bot = stub_bot({"AAA": orb_series()}, longs=["AAA"])
        enable(bot, legacy.ORB_CANDIDATE_TYPE)

        events = bot.check_orb_first_candle_setups()
        assert [event.kind for event in events] == ["candidate"]
        assert "ORB CANDIDATE AAA" in bot.emitted[0][0]["text"]
        assert bot.registered == []

    def test_the_recross_is_informational_and_not_an_outcome(self, monkeypatch):
        stamp = OPEN + timedelta(minutes=5 * 13)
        monkeypatch.setattr(legacy, "get_market_local_now", lambda *a, **k: stamp)
        bot = stub_bot({"AAA": orb_series()}, longs=["AAA"])
        enable(bot, legacy.ORB_LRSI_RECROSS_TYPE)

        events = bot.check_orb_first_candle_setups()
        assert [event.kind for event in events] == ["lrsi_recross"]
        assert bot.registered == []

    def test_each_step_is_switchable_alone(self, at_the_orb_break):
        bot = stub_bot({"AAA": orb_series()}, longs=["AAA"])
        enable(bot, legacy.ORB_CANDIDATE_TYPE)
        # The candidate is enabled but its bar is long past; the break's own
        # type is still off, so this scan says nothing at all.
        assert bot.check_orb_first_candle_setups() == []

    def test_the_same_break_is_not_re_emitted(self, at_the_orb_break):
        bot = stub_bot({"AAA": orb_series()}, longs=["AAA"])
        enable(bot, legacy.ORB_NEW_EXTREME_TYPE)
        bot.check_orb_first_candle_setups()
        assert len(bot.emitted) == 1
        assert bot.check_orb_first_candle_setups() == []
        assert len(bot.emitted) == 1

    def test_a_symbol_with_no_cached_bars_is_silence(self, at_the_orb_break):
        bot = stub_bot({}, longs=["AAA"])
        enable(bot, legacy.ORB_NEW_EXTREME_TYPE)
        assert bot.check_orb_first_candle_setups() == []


class TestTheLane:
    def test_the_ui_reads_both_as_ordinary_m5_alerts(self, at_the_confluence):
        """No D1 routing, no chart-watch privilege, no entry-assist bypass."""
        from ui.models.bounce import (
            BounceAlert,
            is_chart_watch_alert,
            is_entry_assist_text,
        )

        bars = ohlc_bars(decline_then_reversal())
        bot = stub_bot({"AAA": bars}, longs=["AAA"], focus_long=["AAA"])
        enable(bot, legacy.M5_CONFLUENCE_TYPE)
        bot.check_confluence_setups()
        payload, tag = bot.emitted[0]

        alert = BounceAlert.from_callback(payload, tag)
        assert alert.is_d1 is False
        assert is_chart_watch_alert(alert) is False
        assert is_entry_assist_text(alert.raw_text) is False

    def test_every_new_type_has_a_feed_label(self):
        for signal_type in (
            legacy.M5_CONFLUENCE_TYPE,
            legacy.ORB_CANDIDATE_TYPE,
            legacy.ORB_NEW_EXTREME_TYPE,
            legacy.ORB_LRSI_RECROSS_TYPE,
        ):
            assert legacy.BOUNCE_TYPE_LABELS[signal_type]
