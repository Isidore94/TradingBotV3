"""R5 section 3.1 wiring: the LRSI cross engine inside the live detector.

Two things are asserted here that no engine-level test can see.

**The taxonomy pin.** ``BOUNCE_LEARNING_TYPE_KEYS`` is derived from
``BOUNCE_TYPE_DEFAULTS``, so dropping a new engine into that dict would widen
what the learning path treats as an established bounce type - a scoring change
smuggled in as a feature, which ``plan.md`` sec 5 forbids without fixtures.
The M5 signal engines therefore carry their own toggle map, and these tests
fail if anyone ever "tidies" the two together.

**The lane.** R5 section 8.1: a new ``M5_SIGNAL_TAG`` family, never ``d1_flag``,
with no tier bypass and no champion privilege. That is enforced by what the
detector passes to ``gui_callback`` and by the UI's own tag predicates, so it
is checked on both sides of the seam.
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
CHURN_THEN_RUN = [
    100.0, 99.5, 100.2, 99.6, 100.1, 99.4, 100.0, 99.5,
    100.3, 100.9, 101.6, 102.4, 103.3, 104.3, 105.4,
]


@dataclass
class StubBar:
    dt: datetime
    open: float
    high: float
    low: float
    close: float


def bars_from(closes, *, start=OPEN):
    return [
        StubBar(start + timedelta(minutes=5 * index), close, close, close, close)
        for index, close in enumerate(closes)
    ]


def stub_bot(symbol_bars, *, longs=(), shorts=()):
    """A BounceBot with every heavy collaborator replaced by a stub."""
    bot = legacy.BounceBot.__new__(legacy.BounceBot)
    bot._lrsi_cross_state = None
    bot.m5_signal_toggles = dict(legacy.M5_SIGNAL_TYPE_DEFAULTS)
    bot.market_environment_lock = threading.Lock()
    bot.emitted = []
    bot.logged = []

    spy = bars_from(CHURN_THEN_RUN)
    bot._spy_session_bars = lambda: (spy, None)
    bot._watchlist_day_sweep_symbols = lambda side, symbols=None: (
        list(longs) if side == "long" else list(shorts)
    )
    bot.get_cached_5m_bars = lambda symbol: symbol_bars.get(symbol, [])
    bot._log_bounce_candidate_event = lambda *a, **k: {
        "event_id": "evt-1",
        "symbol": a[1] if len(a) > 1 else "",
        "direction": a[2] if len(a) > 2 else "",
    }
    bot._register_bounce_outcome = lambda *a, **k: None
    bot._evaluate_bounce_alert_quality = lambda *a, **k: {"tier": "B"}
    bot._measured_exit_suffix = lambda *a, **k: ""
    bot.log_symbol = lambda symbol, text: bot.logged.append((symbol, text))
    bot.log_bounce_to_file = lambda **k: None
    bot.gui_callback = lambda payload, tag: bot.emitted.append((payload, tag))
    return bot


@pytest.fixture
def at_the_crossing(monkeypatch):
    """Freeze the detector clock where bar 9 is the last completed bar."""
    stamp = OPEN + timedelta(minutes=5 * 10)
    monkeypatch.setattr(legacy, "get_market_local_now", lambda *a, **k: stamp)
    return stamp


class TestTheTaxonomyPin:
    def test_the_new_engines_are_not_learning_types(self):
        for signal_type in legacy.M5_SIGNAL_TYPE_DEFAULTS:
            assert signal_type not in legacy.BOUNCE_TYPE_DEFAULTS
            assert signal_type not in legacy.BOUNCE_LEARNING_TYPE_KEYS

    def test_learning_keys_still_derive_from_the_bounce_defaults_alone(self):
        assert legacy.BOUNCE_LEARNING_TYPE_KEYS == set(legacy.BOUNCE_TYPE_DEFAULTS)

    def test_every_engine_is_labelled_for_the_feed(self):
        for signal_type in legacy.M5_SIGNAL_TYPE_DEFAULTS:
            assert legacy.BOUNCE_TYPE_LABELS[signal_type]

    def test_each_engine_is_switchable_on_its_own(self):
        bot = stub_bot({})
        assert bot.is_m5_signal_enabled(legacy.LRSI_CROSS_20_TYPE)
        bot.set_m5_signal_enabled(legacy.LRSI_CROSS_20_TYPE, False)
        assert not bot.is_m5_signal_enabled(legacy.LRSI_CROSS_20_TYPE)
        assert bot.is_m5_signal_enabled(legacy.LRSI_CROSS_50_TYPE)

    def test_an_unknown_signal_type_is_off_not_on(self):
        assert stub_bot({}).is_m5_signal_enabled("not_a_real_engine") is False


class TestTheSweep:
    def test_a_crossing_on_the_last_completed_bar_alerts_once(self, at_the_crossing):
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])

        assert len(bot.check_lrsi_cross_setups()) == 1
        assert len(bot.emitted) == 1

        # Same bar, next scan cycle: the ledger already holds it.
        assert bot.check_lrsi_cross_setups() == []
        assert len(bot.emitted) == 1

    def test_the_stronger_level_is_the_one_reported(self, at_the_crossing):
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])
        bot.check_lrsi_cross_setups()
        payload, _tag = bot.emitted[0]
        assert "LRSI CROSS 20" in payload["text"]

    def test_a_disabled_engine_emits_nothing(self, at_the_crossing):
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])
        bot.set_m5_signal_enabled(legacy.LRSI_CROSS_20_TYPE, False)

        assert bot.check_lrsi_cross_setups() == []
        assert bot.emitted == []

    def test_a_forming_crossing_bar_is_not_an_alert(self, monkeypatch):
        forming = OPEN + timedelta(minutes=5 * 9 + 4)
        monkeypatch.setattr(legacy, "get_market_local_now", lambda *a, **k: forming)
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])

        assert bot.check_lrsi_cross_setups() == []

    def test_the_short_side_mirrors(self, at_the_crossing):
        mirrored = [200.0 - close for close in CHURN_THEN_RUN]
        bot = stub_bot({"ZZZ": bars_from(mirrored)}, shorts=["ZZZ"])

        assert len(bot.check_lrsi_cross_setups()) == 1
        payload, _tag = bot.emitted[0]
        assert "(short)" in payload["text"]

    def test_a_symbol_with_no_cached_bars_is_silence(self, at_the_crossing):
        bot = stub_bot({}, longs=["AAA"])
        assert bot.check_lrsi_cross_setups() == []

    def test_no_spy_session_means_no_sweep(self, at_the_crossing):
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])
        bot._spy_session_bars = lambda: ([], None)
        assert bot.check_lrsi_cross_setups() == []

    def test_the_state_resets_on_a_new_session(self, at_the_crossing):
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])
        bot.check_lrsi_cross_setups()
        assert len(bot.emitted) == 1

        bot._lrsi_cross_state["date"] = OPEN.date() - timedelta(days=1)
        bot.check_lrsi_cross_setups()
        assert len(bot.emitted) == 2


class TestTheLane:
    def test_the_alert_rides_the_new_tag_family(self, at_the_crossing):
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])
        bot.check_lrsi_cross_setups()
        _payload, tag = bot.emitted[0]

        assert tag == M5_SIGNAL_TAG == "m5_signal"
        assert not tag.startswith("d1_flag")

    def test_the_ui_reads_it_as_an_ordinary_m5_alert(self, at_the_crossing):
        """No D1 routing, no chart-watch privilege, no entry-assist bypass."""
        from ui.models.bounce import (
            BounceAlert,
            is_chart_watch_alert,
            is_entry_assist_text,
        )

        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])
        bot.check_lrsi_cross_setups()
        payload, tag = bot.emitted[0]

        alert = BounceAlert.from_callback(payload, tag)
        assert alert.is_d1 is False
        assert is_chart_watch_alert(alert) is False
        assert is_entry_assist_text(alert.raw_text) is False
