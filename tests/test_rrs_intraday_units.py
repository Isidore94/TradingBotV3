"""Intraday RRS on the rolling engine: the threshold keys and each converted constant.

Every desk-scale RRS constant is multiplied by ``rrs_config.DESK_TO_ROLLING``
(0.5) on the rolling engine; the score bonus keeps its cap and its points at
the cutoff. Each test states both engines so the desk numbers stay pinned.
"""

from __future__ import annotations

import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bounce_bot  # noqa: E402

ROLLING = "rolling_hourly"
DESK = "desk"


@pytest.fixture(params=[ROLLING, DESK])
def engine(request, monkeypatch):
    monkeypatch.setenv("TRADINGBOTV3_RRS_ENGINE", request.param)
    return request.param


def _bot():
    bot = bounce_bot.BounceBot.__new__(bounce_bot.BounceBot)
    bot.rrs_lock = threading.Lock()
    bot.rrs_threshold = 2.0  # the trader's saved desk-scale number
    bot.rrs_bar_size = "5 mins"
    bot.rrs_duration = "5 D"
    bot.rrs_length = 12
    bot.rrs_timeframe_key = "5m"
    return bot


# --- threshold keys ---------------------------------------------------------


def test_the_saved_desk_threshold_is_never_read_on_the_rolling_scale(monkeypatch):
    monkeypatch.setenv("TRADINGBOTV3_RRS_ENGINE", ROLLING)
    bot = _bot()
    assert bot.get_rrs_settings()[0] == 1.0
    bot.set_rrs_threshold(1.25)
    assert bot.rolling_rrs_threshold == 1.25
    assert bot.rrs_threshold == 2.0, "the desk number is left alone"
    assert bot.get_rrs_settings()[0] == 1.25


def test_the_desk_engine_keeps_its_own_threshold(monkeypatch):
    monkeypatch.setenv("TRADINGBOTV3_RRS_ENGINE", DESK)
    bot = _bot()
    bot.rolling_rrs_threshold = 1.25
    assert bot.get_rrs_settings()[0] == 2.0
    bot.set_rrs_threshold(2.5)
    assert bot.rrs_threshold == 2.5
    assert bot.rolling_rrs_threshold == 1.25


def test_the_child_process_reports_both_thresholds():
    from ui.services import bounce_process

    bot = _bot()
    bot.rolling_rrs_threshold = 0.9
    state = bounce_process._state(bot)
    assert state["rrs_threshold"] == 2.0
    assert state["rolling_rrs_threshold"] == 0.9


def test_the_service_mirrors_the_active_engines_threshold(engine):
    from ui.services import bounce_service

    class _Bot:
        rrs_threshold = 2.4
        rolling_rrs_threshold = 1.1

    expected = 1.1 if engine == ROLLING else 2.4
    assert bounce_service.bot_rrs_threshold(_Bot(), 9.0) == expected
    assert bounce_service.bot_rrs_threshold(object(), 9.0) == 9.0


def test_the_settings_spinbox_steps_finely_enough_for_the_rolling_scale(monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from ui.panels.settings_panel import SettingsPanel
    from ui.state import UiState

    QApplication.instance() or QApplication([])

    class _Service:
        rrs_threshold = 1.05
        rrs_timeframe_key = "5m"
        bounce_type_settings: dict = {}

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    panel = SettingsPanel(UiState(), bounce_service=_Service())
    try:
        spin = panel.rrs_threshold_input
        assert spin.singleStep() <= 0.05
        assert spin.decimals() >= 2
        assert spin.value() == pytest.approx(1.05)
    finally:
        panel.deleteLater()


# --- converted constants ------------------------------------------------------


def _score(bot, direction, **context):
    base = {"rrs_spy": "", "rrs_sector": "", "rrs_industry": ""}
    base.update(context)
    return bot._score_bounce_candidate_snapshot(direction, {"vwap": {}}, base)


def test_the_spy_bonus_pays_the_same_points_at_the_cutoff(engine):
    bot = _bot()
    cutoff = 1.0 if engine == ROLLING else 2.0
    base = _score(bot, "long")
    assert _score(bot, "long", rrs_spy=cutoff) == base + 4.0
    assert _score(bot, "short", rrs_spy=-cutoff) == base + 4.0
    assert _score(bot, "long", rrs_spy=3.0 * cutoff) == base + 12.0
    assert _score(bot, "long", rrs_spy=10.0) == base + 12.0, "cap stays 12"
    assert _score(bot, "long", rrs_spy=-cutoff) == base


def test_the_industry_bonus_keeps_its_floor_and_cap(engine):
    bot = _bot()
    unit = 0.5 if engine == ROLLING else 1.0
    base = _score(bot, "long")
    assert _score(bot, "long", rrs_industry=3.0 * unit) == base + 6.0
    assert _score(bot, "long", rrs_industry=0.5 * unit) == base + 4.0, "floor stays 4"
    assert _score(bot, "long", rrs_industry=9.0) == base + 10.0, "cap stays 10"


def test_the_focus_gate_minimum_is_on_the_active_scale(engine):
    bot = _bot()
    threshold = 1.0 if engine == ROLLING else 2.0
    floor = 1.375 if engine == ROLLING else 2.75  # max(2.75 x scale, cutoff + 0.75 x scale)

    def significant(value):
        entry = {"rrs": value, "move_ratio": 1.0, "excess_move_ratio": 0.6}
        return bot._focus_rrs_is_significant(entry, threshold)

    assert not significant(floor - 0.01)
    assert significant(floor + 0.01)


def test_the_impulse_counter_read_is_on_the_active_scale(engine, monkeypatch):
    bot = _bot()
    bot.get_market_environment = lambda: "bullish_strong"
    counter = 0.175 if engine == ROLLING else 0.35

    def profile(recent_rrs):
        pre = [{"rrs": 1.0, "move_ratio": 1.0, "excess_move_ratio": 0.5} for _ in range(4)]
        recent = [{"rrs": recent_rrs, "move_ratio": -0.5, "excess_move_ratio": -0.2} for _ in range(3)]
        return pre + recent

    assert bot._impulse_regime_transition_ok("X", "long", profile(-(counter + 0.01)))
    assert not bot._impulse_regime_transition_ok("X", "long", profile(-(counter - 0.01)))


def test_alignment_buckets_follow_each_rows_own_scale(engine):
    """Stored rows keep their scale whatever engine runs now."""
    align = bounce_bot._bounce_rrs_alignment
    assert align({"direction": "long", "rrs_spy": 1.2}) == "aligned"
    assert align({"direction": "long", "rrs_spy": 2.0}) == "strong_aligned"
    tagged = {"direction": "long", "rrs_spy": 1.2, "rrs_engine": ROLLING}
    assert align(tagged) == "strong_aligned"
    assert align({**tagged, "rrs_spy": 0.9}) == "aligned"
    assert align({"direction": "short", "rrs_spy": -1.0, "rrs_engine": ROLLING}) == "strong_aligned"


def test_the_environment_context_floor_is_a_fraction_of_the_active_threshold(engine):
    """0.35 x threshold: scale-free, so it moves with the cutoff by itself."""
    bot = _bot()
    threshold = 1.0 if engine == ROLLING else 2.0
    window = {"spy_weak": True}

    def hit(value):
        entry = {"rrs": value, "move_ratio": 0.0, "excess_move_ratio": 0.5}
        return bot._profile_item_matches_environment_context(entry, "long", window, threshold)

    assert hit(0.35 * threshold + 0.01)
    assert not hit(0.35 * threshold - 0.01)


def test_the_bounce_context_carries_the_engine(engine):
    bot = _bot()
    bot.latest_rrs_payload = {"timeframe_key": "5m", "results": [], "rrs_all": {"ABC": (1.2, 0.3)}}
    bot.symbol_classification_cache = {}
    bot.sector_etf_map = {}
    bot.latest_market_internals = None
    bot.get_market_environment = lambda: "bullish_strong"
    bot._human_focus_side_for_symbol = lambda symbol, direction=None: ""
    bot.longs, bot.shorts, bot.auto_longs, bot.auto_shorts = ["ABC"], [], [], []
    bot.session_rvol_for = lambda symbol: None
    context = bot._build_bounce_context_snapshot("ABC", "long")
    assert context["rrs_engine"] == engine
    assert context["rrs_spy"] == 1.2


# --- review learning -----------------------------------------------------------


def test_review_learning_flat_band_follows_the_rows_scale():
    from review_learning import Episode, _rrs_alignment

    def bucket(value, engine=""):
        return _rrs_alignment(Episode(trade_date="2026-09-28", symbol="X", side="LONG", rrs_spy=value, rrs_engine=engine))

    assert bucket(0.4) == "flat"
    assert bucket(0.4, ROLLING) == "aligned"
    assert bucket(0.2, ROLLING) == "flat"
    assert bucket(-0.3, ROLLING) == "against"


def test_review_events_and_scoreboard_lift_the_engine_tag():
    import review_events
    import setup_scoreboard

    assert "rrs_engine" in review_events._CONTEXT_FIELDS
    assert "rrs_engine" in setup_scoreboard.CONTEXT_FIELDS


# --- group tape ----------------------------------------------------------------

TAPE_DAY = datetime(2026, 8, 27, 6, 30)


def _tape_rows(steps, *, base, days=5, per_day=78, spread=0.2):
    """Multi-day 5m rows: flat history, then ``steps`` added per bar today."""
    rows = []
    for day in range(days):
        start = TAPE_DAY - timedelta(days=days - 1 - day)
        price = base
        for index in range(per_day if day < days - 1 else len(steps)):
            if day == days - 1:
                price += steps[index]
            wobble = spread * (1.0 + 0.3 * (index % 3))
            rows.append(
                {
                    "dt": start + timedelta(minutes=5 * index),
                    "open": price,
                    "high": price + wobble,
                    "low": price - wobble,
                    "close": price,
                }
            )
    return rows


def test_rolling_tape_chips_read_today_only_after_the_first_hour(monkeypatch):
    monkeypatch.setenv("TRADINGBOTV3_RRS_ENGINE", ROLLING)
    import group_rrs

    spy = _tape_rows([0.0] * 40, base=500.0, spread=0.5)
    leader = _tape_rows([0.08] * 40, base=100.0)

    def chips(bars_today):
        now = TAPE_DAY + timedelta(minutes=5 * bars_today)
        return group_rrs.rolling_rrs_windows(
            leader[: 4 * 78 + bars_today], spy[: 4 * 78 + bars_today], now=now
        )

    need = group_rrs.rolling_minimum_bars_for("30")
    assert need == 15
    assert all(value is None for value in chips(need - 1).values())
    ready = chips(need)
    assert all(value is not None and value > 1.0 for value in ready.values())
    late = chips(40)
    assert late["30"] > 1.0 and late["90"] > 1.0


def test_rolling_tape_fetches_five_days(monkeypatch):
    pytest.importorskip("PySide6")
    from ui.services import group_tape_service

    monkeypatch.setenv("TRADINGBOTV3_RRS_ENGINE", ROLLING)
    assert group_tape_service.fetch_period() == "5d"
    monkeypatch.setenv("TRADINGBOTV3_RRS_ENGINE", DESK)
    assert group_tape_service.fetch_period() == "1d"


def test_the_desk_tape_is_untouched(monkeypatch):
    monkeypatch.setenv("TRADINGBOTV3_RRS_ENGINE", ROLLING)
    import group_rrs

    assert group_rrs.minimum_bars_for("30") == 8
    assert group_rrs.RRS_WINDOWS == {"30": 6, "60": 12, "90": 18}
