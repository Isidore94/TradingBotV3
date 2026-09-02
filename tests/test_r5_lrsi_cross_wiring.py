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

**The retirement (2026-09-01).** ``LRSI_M5_ALERTS_RETIRED`` silences the GUI
leg only. Every assertion about detection, the candidate row, the outcome
registration and the tier is unchanged; the message these tests used to read
off ``gui_callback`` is now read off the ``LEARNING_ONLY`` line, which carries
the same text. The lane tests build the payload themselves for the same
reason - the tag family still has to be right the day the trader un-retires it.
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


def learning_only_messages(bot):
    """The alert text the trader no longer hears, off the LEARNING_ONLY line.

    Retired 2026-09-01, so ``bot.emitted`` is empty by design; the message
    itself is unchanged and is what these tests are actually about.
    """
    marker = "LEARNING_ONLY [LRSI M5 retired]: "
    return [text[len(marker):] for _symbol, text in bot.logged if text.startswith(marker)]


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
    def test_a_crossing_on_the_last_completed_bar_fires_once(self, at_the_crossing):
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])

        assert len(bot.check_lrsi_cross_setups()) == 1
        assert len(learning_only_messages(bot)) == 1

        # Same bar, next scan cycle: the ledger already holds it.
        assert bot.check_lrsi_cross_setups() == []
        assert len(learning_only_messages(bot)) == 1

    def test_the_stronger_level_is_the_one_reported(self, at_the_crossing):
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])
        bot.check_lrsi_cross_setups()
        assert "LRSI CROSS 20" in learning_only_messages(bot)[0]

    def test_a_disabled_engine_detects_nothing_at_all(self, at_the_crossing):
        """The toggle gates DETECTION, which is why the retirement is not
        applied here: a False toggle drops the event before the candidate row
        and the outcome registration, and the trader asked to keep those."""
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])
        bot.set_m5_signal_enabled(legacy.LRSI_CROSS_20_TYPE, False)

        assert bot.check_lrsi_cross_setups() == []
        assert bot.emitted == []
        assert learning_only_messages(bot) == []

    def test_a_forming_crossing_bar_is_not_an_alert(self, monkeypatch):
        forming = OPEN + timedelta(minutes=5 * 9 + 4)
        monkeypatch.setattr(legacy, "get_market_local_now", lambda *a, **k: forming)
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])

        assert bot.check_lrsi_cross_setups() == []

    def test_the_short_side_mirrors(self, at_the_crossing):
        mirrored = [200.0 - close for close in CHURN_THEN_RUN]
        bot = stub_bot({"ZZZ": bars_from(mirrored)}, shorts=["ZZZ"])

        assert len(bot.check_lrsi_cross_setups()) == 1
        assert "(short)" in learning_only_messages(bot)[0]

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
        assert len(learning_only_messages(bot)) == 1

        bot._lrsi_cross_state["date"] = OPEN.date() - timedelta(days=1)
        bot.check_lrsi_cross_setups()
        assert len(learning_only_messages(bot)) == 2


class TestTheLane:
    """The lane the engine WOULD ride if the retirement were lifted.

    R1: these had become assertions about a payload the tests built themselves,
    which is close to a tautology - `M5_SIGNAL_TAG == "m5_signal"` would pass
    with the emit path deleted. They now monkeypatch `LRSI_M5_ALERTS_RETIRED`
    to False and assert the REAL emit, so un-retiring the engine is a
    one-constant change whose consequences are already pinned. That matters
    because the trader retired the alerts and kept the evidence explicitly to
    decide later; the lane has to still work when they do.
    """

    def test_the_alert_would_ride_the_new_tag_family(self, at_the_crossing, monkeypatch):
        monkeypatch.setattr(legacy, "LRSI_M5_ALERTS_RETIRED", False)
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])
        bot.check_lrsi_cross_setups()

        assert bot.emitted, "un-retired, the engine must reach the GUI"
        tags = {tag for _payload, tag in bot.emitted}
        assert tags == {M5_SIGNAL_TAG} == {"m5_signal"}
        assert not any(str(tag).startswith("d1_flag") for tag in tags)

    def test_the_ui_reads_it_as_an_ordinary_m5_alert(self, at_the_crossing, monkeypatch):
        """No D1 routing, no chart-watch privilege, no entry-assist bypass."""
        from ui.models.bounce import (
            BounceAlert,
            is_chart_watch_alert,
            is_entry_assist_text,
        )

        monkeypatch.setattr(legacy, "LRSI_M5_ALERTS_RETIRED", False)
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])
        bot.check_lrsi_cross_setups()

        payload, tag = bot.emitted[0]
        alert = BounceAlert.from_callback(payload, tag)
        assert alert.is_d1 is False
        assert is_chart_watch_alert(alert) is False
        assert is_entry_assist_text(alert.raw_text) is False


# ==========================================================================
# Retired 2026-09-01 - trader: "LRSI alerts seem to be mostly spam. however I
# enjoy them as something that can boost the potential of an alert. for now
# let's put them on the back burner. let's measure how they perform on
# different timeframes but no need for their M5 alerts."
#
# The whole point is that ONLY the GUI leg goes. Detection, the candidate row,
# `intraday_bounce_outcomes.csv`, the learning tier and the PROVEN stamp all
# keep running, because the 'different timeframes' measurement the trader
# asked for is built on exactly those rows.
# ==========================================================================
class TestTheM5AlertRetirement:
    def test_the_flag_is_on(self):
        assert legacy.LRSI_M5_ALERTS_RETIRED is True

    def test_a_crossing_logs_an_outcome_row_and_produces_no_gui_callback(
        self, at_the_crossing
    ):
        """Fail-before-fix: on the un-fixed code `bot.emitted` has one entry."""
        bot = stub_bot({"AAA": _ranged_bars(CHURN_THEN_RUN)}, longs=["AAA"])
        registered = []
        rows = []
        tiers = []
        bot._register_bounce_outcome = lambda *a, **k: registered.append(a)
        bot._log_bounce_candidate_event = lambda *a, **k: (
            rows.append(a) or {"event_id": "evt-1", "symbol": a[1], "direction": a[2]}
        )
        bot.record_alert_tier = lambda event_id, quality: tiers.append((event_id, quality))

        assert len(bot.check_lrsi_cross_setups()) == 1

        assert bot.emitted == [], "the retired engine must never reach the GUI"
        assert len(registered) == 1, "the outcome row is the evidence being kept"
        assert len(rows) == 1, "the candidate row is still logged"
        assert len(tiers) == 1, "the learning tier is still recorded"
        assert len(learning_only_messages(bot)) == 1

    def test_the_simplified_bounce_log_still_gets_the_row(self, at_the_crossing):
        """`journal_analytics.AutoTagger` reads INTRADAY_BOUNCES_CSV to answer
        which of my setups a trade was; losing it would blank the tag on a real
        LRSI trade. This is where the retirement differs from H1's."""
        bot = stub_bot({"AAA": bars_from(CHURN_THEN_RUN)}, longs=["AAA"])
        filed = []
        bot.log_bounce_to_file = lambda **k: filed.append(k)

        bot.check_lrsi_cross_setups()

        assert len(filed) == 1
        assert set(filed[0]["levels"]) == {legacy.LRSI_CROSS_20_TYPE}

    def test_the_detection_toggles_stay_on(self):
        """Flipping these False would stop DETECTION and therefore the evidence
        (`check_lrsi_cross_setups` drops the event before the candidate row),
        which is the opposite of what the trader asked for."""
        assert legacy.M5_SIGNAL_TYPE_DEFAULTS[legacy.LRSI_CROSS_20_TYPE] is True
        assert legacy.M5_SIGNAL_TYPE_DEFAULTS[legacy.LRSI_CROSS_50_TYPE] is True


# ==========================================================================
# R10.B / audit D5a - the synthetic flat bar produced ZERO outcome rows
#
# `_emit_lrsi_cross_alert` built one bar dict with open=high=low=close and
# passed it everywhere. `_build_bounce_trade_plan` takes a long's stop from the
# bounce bar's LOW, so stop == entry, risk == 0, and `_register_bounce_outcome`
# returned at its `risk_per_share == ""` guard. Measured over the audit window:
# 0 outcome rows for `lrsi_cross_20` and `lrsi_cross_50`.
#
# The fix hands the REAL signal bar to the outcome registration ONLY. The alert
# row, the tier evaluation and the message keep the flat bar they always had -
# those feed `_evaluate_bounce_alert_quality`, and moving them would be a
# scoring change smuggled in as an evidence fix.
# ==========================================================================
def _ranged_bars(closes, *, start=OPEN, spread=0.4):
    """Bars with a real range, so a stop taken from the low is not the close."""
    return [
        StubBar(
            start + timedelta(minutes=5 * index),
            close - spread / 2,
            close + spread,
            close - spread,
            close,
        )
        for index, close in enumerate(closes)
    ]


def _recording_bot(symbol_bars, **kwargs):
    bot = stub_bot(symbol_bars, **kwargs)
    bot.registered = []
    bot._register_bounce_outcome = lambda *a, **k: bot.registered.append((a, k))
    return bot


def test_the_outcome_registration_gets_the_real_signal_bar(at_the_crossing):
    """Fail-before-fix (D5a). The bar handed to outcome registration must have
    the signal bar's own high and low, or no stop can be derived from it."""
    bars = _ranged_bars(CHURN_THEN_RUN)
    bot = _recording_bot({"AAA": bars}, longs=["AAA"])

    bot.check_lrsi_cross_setups()

    assert bot.registered, "the crossing must still register an outcome"
    args, _kwargs = bot.registered[0]
    bounce_candle = args[3]
    assert bounce_candle["high"] != bounce_candle["low"], (
        "a flat bar makes stop == entry, risk == 0, and registration returns early"
    )
    signal_bar = bars[9]
    assert bounce_candle["high"] == pytest.approx(signal_bar.high)
    assert bounce_candle["low"] == pytest.approx(signal_bar.low)
    assert bounce_candle["close"] == pytest.approx(signal_bar.close)


def test_a_registerable_risk_now_exists_for_an_lrsi_cross(at_the_crossing):
    """The consequence that matters: the plan the registration builds carries a
    positive risk, so the row is no longer discarded at the guard."""
    bars = _ranged_bars(CHURN_THEN_RUN)
    bot = _recording_bot({"AAA": bars}, longs=["AAA"])
    bot.atr_cache = {}
    bot._to_float_or_blank = legacy.BounceBot._to_float_or_blank.__get__(bot)

    bot.check_lrsi_cross_setups()
    args, _kwargs = bot.registered[0]
    _symbol, side, levels, bounce_candle, current_candle = args[0], args[1], args[2], args[3], args[4]

    plan = legacy.BounceBot._build_bounce_trade_plan(
        bot, side, levels, bounce_candle, current_candle, symbol="AAA"
    )
    assert plan["risk_per_share"] != ""
    assert float(plan["risk_per_share"]) > 0


def test_the_alert_row_and_the_tier_still_see_the_flat_bar(at_the_crossing):
    """The scoring boundary. `_evaluate_bounce_alert_quality` reads the row
    built from the flat bar; widening that bar would move tiers, which is a
    scoring change and needs fixtures, not an evidence packet."""
    seen = []
    bars = _ranged_bars(CHURN_THEN_RUN)
    bot = _recording_bot({"AAA": bars}, longs=["AAA"])
    bot._log_bounce_candidate_event = lambda *a, **k: (
        seen.append(a[5]) or {"event_id": "evt-1", "symbol": a[1], "direction": a[2]}
    )

    bot.check_lrsi_cross_setups()

    assert seen, "the alert row is still logged"
    flat = seen[0]
    assert flat["open"] == flat["high"] == flat["low"] == flat["close"]


# ---------------------------------------------------------------------------
# Decision B.2 (2026-08-25): the recovered bar must BE the event's bar
# ---------------------------------------------------------------------------
def _shifted_cache_bot(bars):
    from types import SimpleNamespace  # noqa: F401  (used by the callers below)

    bot = legacy.BounceBot.__new__(legacy.BounceBot)
    bot.get_cached_5m_bars = lambda symbol: bars
    return bot


def test_a_shifted_cache_with_duplicate_closes_does_not_recover_the_wrong_bar():
    """Sol C3, reproduced verbatim.

    The close match alone is not an identity. A cache that has shifted by one
    bar, where two adjacent bars happen to close at the same price, returned the
    06:30 bar for a 06:35 event - and a wrong bar is worse than no bar, because
    it produces a plausible stop from the wrong price. The event's own
    `bar_time` now has to match too.
    """
    from types import SimpleNamespace

    bars = [
        SimpleNamespace(dt=datetime(2026, 8, 25, 6, 30), open=98, high=103, low=97, close=100),
        SimpleNamespace(dt=datetime(2026, 8, 25, 6, 35), open=99, high=101, low=98, close=100),
    ]
    bot = _shifted_cache_bot(bars)
    fallback = {"time": "20260825  06:35:00", "open": 100, "high": 100, "low": 100, "close": 100}

    got = bot._signal_bar_dict("AAA", 0, fallback, bar_time=datetime(2026, 8, 25, 6, 35))

    assert got is fallback, "a shifted cache recovered the 06:30 bar for a 06:35 event"


def test_the_matching_bar_is_still_recovered():
    """The repair must not cost the recovery it exists for (D5a)."""
    from types import SimpleNamespace

    bars = [
        SimpleNamespace(dt=datetime(2026, 8, 25, 6, 30), open=98, high=103, low=97, close=100),
        SimpleNamespace(dt=datetime(2026, 8, 25, 6, 35), open=99, high=101, low=98, close=100),
    ]
    bot = _shifted_cache_bot(bars)
    fallback = {"time": "20260825  06:35:00", "open": 100, "high": 100, "low": 100, "close": 100}

    got = bot._signal_bar_dict("AAA", 1, fallback, bar_time=datetime(2026, 8, 25, 6, 35))

    assert got is not fallback
    assert got["high"] == 101 and got["low"] == 98


def test_an_event_with_no_bar_time_falls_back_rather_than_guessing():
    """Missing data is uncertainty, never confirmation (plan.md sec 5). With no
    stamp to check against there is no way to know the index still points at the
    event's bar, and the flat fallback is at least the event's own prices."""
    from types import SimpleNamespace

    bars = [SimpleNamespace(dt=datetime(2026, 8, 25, 6, 35), open=99, high=101, low=98, close=100)]
    bot = _shifted_cache_bot(bars)
    fallback = {"time": "", "open": 100, "high": 100, "low": 100, "close": 100}

    assert bot._signal_bar_dict("AAA", 0, fallback, bar_time=None) is fallback
