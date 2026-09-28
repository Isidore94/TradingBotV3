"""Golden results for the intraday RRS paths of the M5 bounce bot.

Recorded on the code BEFORE the rolling-RRS switch (commit pinned below), so
the desk engine (``TRADINGBOTV3_RRS_ENGINE=desk``) must keep reproducing it.
Five synthetic sessions of 78 M5 bars feed the real ``run_rrs_scan`` (5m, 15m,
1h payloads), the bounce score bonus, the focus gate, the alignment buckets,
the impulse profile and the M5 group strength.

Nothing here touches a live store: all bars are synthetic and every I/O seam
is stubbed.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bounce_bot  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "rrs_intraday_golden.json"

# The commit the fixture was recorded at (the rrs_config base, before any
# intraday call site moved to the rolling RRS). Change the code, not the pin.
PINNED_SOURCE_COMMIT = "237f3ac"

SESSION_DATES = (
    date(2026, 6, 1),
    date(2026, 6, 2),
    date(2026, 6, 3),
    date(2026, 6, 4),
    date(2026, 6, 5),
)
SESSION_OPEN_HOUR = 6
SESSION_OPEN_MINUTE = 30
BARS_PER_SESSION = 78

SPY = "SPY"
SECTOR_ETF = "XLK"
INDUSTRY_ETF = "SMH"
SECTOR_ETF_MAP = {"technology": SECTOR_ETF}
INDUSTRY_ETF_MAP_FILE = {
    "yahoo_industryKey_to_ref": {
        "semiconductors": {"sectorKey": "technology", "etf": INDUSTRY_ETF},
    }
}
STOCKS = ("STRONG", "WEAK", "BURST", "FLAT")
LONGS = ["STRONG", "BURST", "FLAT"]
SHORTS = ["WEAK"]
GUI_TIMEFRAME_KEY = "5m"
CYCLE_TIMEFRAMES = ("5m", "15m", "1h")

# symbol -> (seed, start price, beta to SPY, last-day alpha per bar in noise units)
RECIPE = {
    SPY: (101, 500.0, 0.0, 0.0),
    "STRONG": (102, 120.0, 1.0, 2.0),
    "WEAK": (103, 80.0, 1.0, -2.0),
    "BURST": (104, 60.0, 1.0, 0.0),  # burst for the first 18 bars of the last day
    "FLAT": (105, 45.0, 1.0, 0.0),
    SECTOR_ETF: (106, 240.0, 1.0, 0.10),
    INDUSTRY_ETF: (107, 270.0, 1.0, 0.15),
}
BURST_BARS = 18
BURST_ALPHA = 1.2
NOISE_PCT = 0.0012  # per-bar noise, fraction of price


def _bar_times():
    times = []
    for session_date in SESSION_DATES:
        start = datetime(
            session_date.year, session_date.month, session_date.day, SESSION_OPEN_HOUR, SESSION_OPEN_MINUTE
        )
        times.extend(start + timedelta(minutes=5 * i) for i in range(BARS_PER_SESSION))
    return times


def _spy_returns():
    rng = random.Random(RECIPE[SPY][0])
    out = []
    for index in range(len(SESSION_DATES) * BARS_PER_SESSION):
        last_day = index >= (len(SESSION_DATES) - 1) * BARS_PER_SESSION
        drift = 0.25 if last_day else 0.02
        out.append((drift + rng.gauss(0.0, 1.0)) * NOISE_PCT)
    return out


def _series(symbol):
    seed, price, beta, alpha = RECIPE[symbol]
    rng = random.Random(seed)
    spy_returns = _spy_returns()
    last_day_start = (len(SESSION_DATES) - 1) * BARS_PER_SESSION
    bars = []
    for index, dt in enumerate(_bar_times()):
        if symbol == SPY:
            ret = spy_returns[index]
        else:
            ret = beta * spy_returns[index] + rng.gauss(0.0, 1.0) * NOISE_PCT
            if index >= last_day_start:
                ret += alpha * NOISE_PCT
                if symbol == "BURST" and index - last_day_start < BURST_BARS:
                    ret += BURST_ALPHA * NOISE_PCT
        open_ = price
        close = round(open_ * (1.0 + ret), 4)
        span = open_ * NOISE_PCT * 0.8
        high = round(max(open_, close) + rng.uniform(0.1, 1.0) * span, 4)
        low = round(min(open_, close) - rng.uniform(0.1, 1.0) * span, 4)
        bars.append(
            bounce_bot.IbBar(
                dt=dt, open=round(open_, 4), high=high, low=low, close=close, volume=float(10_000 + index)
            )
        )
        price = close
    return bars


def universe(last_day_bars=BARS_PER_SESSION):
    """symbol -> bars; ``last_day_bars`` truncates the final session."""
    keep = (len(SESSION_DATES) - 1) * BARS_PER_SESSION + last_day_bars
    return {symbol: _series(symbol)[:keep] for symbol in RECIPE}


def _fake_session_open_naive(reference=None, local_timezone_name=None):
    ref = reference if isinstance(reference, datetime) else datetime(2026, 6, 1, 9, 0)
    return datetime(ref.year, ref.month, ref.day, SESSION_OPEN_HOUR, SESSION_OPEN_MINUTE)


def _set_universe(bot, bars_by_symbol):
    bot.latest_bars = {}
    for symbol, bars in bars_by_symbol.items():
        bot.latest_bars[f"{symbol}|5 D|5 mins"] = bars
        bot.latest_bars.setdefault(symbol, bars)


def make_bot(bars_by_symbol, *, monkeypatch):
    """A BounceBot whose RRS maths is real and whose I/O is stubbed."""
    monkeypatch.setattr(bounce_bot, "get_market_session_open_naive", _fake_session_open_naive)
    monkeypatch.setattr(bounce_bot, "_load_industry_etf_map_file", lambda: INDUSTRY_ETF_MAP_FILE)
    monkeypatch.setattr(
        bounce_bot, "load_and_update_industry_etf_map", lambda *args, **kwargs: INDUSTRY_ETF_MAP_FILE
    )
    bot = object.__new__(bounce_bot.BounceBot)
    bot.rrs_lock = threading.Lock()
    bot.rrs_threshold = bounce_bot.RRS_DEFAULT_THRESHOLD
    bot.rrs_length = bounce_bot.RRS_LENGTH
    bot.rrs_bar_size = bounce_bot.RRS_TIMEFRAMES[GUI_TIMEFRAME_KEY]["bar_size"]
    bot.rrs_duration = bounce_bot.RRS_TIMEFRAMES[GUI_TIMEFRAME_KEY]["duration"]
    bot.rrs_timeframe_key = GUI_TIMEFRAME_KEY
    bot.longs = list(LONGS)
    bot.shorts = list(SHORTS)
    bot.auto_longs = []
    bot.auto_shorts = []
    bot.master_avwap_d1_watchlist = {}
    bot.master_avwap_d1_upgrade_alerts = {}
    bot.master_avwap_focus_map = {}
    bot.sector_etf_map = dict(SECTOR_ETF_MAP)
    bot.industry_map_data = INDUSTRY_ETF_MAP_FILE
    bot.symbol_classification_cache = {
        symbol: {
            "symbol": symbol,
            "sectorKey": "technology",
            "sector": "Technology",
            "industryKey": "semiconductors",
            "industry": "Semiconductors",
        }
        for symbol in STOCKS
    }
    bot.latest_scan_extremes = {}
    bot.latest_rrs_payload = None
    bot.latest_market_internals = None
    bot.earnings_reaction_filter_cache = {}
    _set_universe(bot, bars_by_symbol)
    bot.request_historical_bars = Mock(return_value=[])
    bot.load_master_avwap_focus = Mock()
    bot.load_master_avwap_d1_watchlist = Mock()
    bot.load_master_avwap_d1_upgrade_alerts = Mock()
    bot.load_master_avwap_d1_zone_arms = Mock()
    bot.get_master_avwap_d1_watch_symbols = Mock(return_value=[])
    bot._human_focus_symbols = Mock(return_value=set())
    bot._human_focus_side_for_symbol = Mock(return_value="")
    bot._chart_watch_symbols = Mock(return_value=set())
    bot._load_earnings_reaction_symbols = Mock(return_value=set())
    bot._symbol_matches_alias_set = Mock(return_value=False)
    bot._environment_scan_is_active = Mock(return_value=True)
    bot._record_environment_focus_history = Mock()
    bot._log_scan_extremes = Mock()
    bot._log_group_strength_extremes = Mock()
    bot._emit_master_avwap_focus_rrs_alerts = Mock()
    bot.get_market_environment = Mock(return_value="bullish_strong")
    bot.compute_group_strengths = Mock(return_value={"M5": {"sectors": [], "industries": []}})
    bot.compute_market_internals = Mock(return_value={"as_of": "stub", "rows": []})
    bot.session_rvol_for = Mock(return_value=None)
    bot.gui_callback = None
    return bot


# --- canonical form -------------------------------------------------------


def _jsonable(value):
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, date):
        return {"__date__": value.isoformat()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    return value


def payload_body(payload):
    """The payload minus its wall-clock timestamp and the engine tag."""
    body = {key: value for key, value in dict(payload).items() if key not in ("timestamp", "rrs_engine")}
    return _jsonable(body)


PROBE_LEVELS = {"vwap": 1.0, "ema_20": 1.0}
FOCUS_PROBES = (0.5, 1.2, 1.4, 1.5, 2.0, 2.74, 2.76, 3.5, -1.4, -2.76, -3.5)
ALIGNMENT_PROBES = (-2.5, -2.0, -1.99, -1.0, -0.4, 0.0, 0.4, 1.0, 1.99, 2.0, 2.5)


def record(monkeypatch):
    """Every recorded output, keyed by what produced it."""
    bot = make_bot(universe(), monkeypatch=monkeypatch)
    bot.run_rrs_scan()
    payloads = {key: payload_body(bot.rrs_payload_for(key)) for key in CYCLE_TIMEFRAMES}

    scores = {}
    bot.latest_rrs_payload = bot.rrs_payload_for("5m")
    for symbol in STOCKS:
        for direction in ("long", "short"):
            context = bot._build_bounce_context_snapshot(symbol, direction)
            scores[f"{symbol}|{direction}"] = {
                "rrs_spy": context.get("rrs_spy"),
                "rrs_sector": context.get("rrs_sector"),
                "rrs_industry": context.get("rrs_industry"),
                "score": bot._score_bounce_candidate_snapshot(direction, PROBE_LEVELS, context),
            }

    focus = {}
    for value in FOCUS_PROBES:
        entry = {"rrs": value, "move_ratio": 1.0 if value > 0 else -1.0, "excess_move_ratio": 0.6 if value > 0 else -0.6}
        focus[repr(value)] = bool(bot._focus_rrs_is_significant(entry, bot.rrs_threshold))

    alignment = {}
    for value in ALIGNMENT_PROBES:
        for direction in ("long", "short"):
            alignment[f"{value!r}|{direction}"] = bounce_bot._bounce_rrs_alignment(
                {"direction": direction, "rrs_spy": value}
            )

    bars = universe()
    last_day = SESSION_DATES[-1]
    impulse = {}
    for symbol in STOCKS:
        today_sym = [bar for bar in bars[symbol] if bar.dt.date() == last_day]
        today_spy = [bar for bar in bars[SPY] if bar.dt.date() == last_day]
        profile = bot._build_intraday_rrs_profile(today_sym, today_spy, length=bounce_bot.IMPULSE_RRS_PROFILE_LENGTH)
        impulse[symbol] = {
            "profile": _jsonable(profile),
            "long_ok": bool(bot._impulse_regime_transition_ok(symbol, "long", profile)),
            "short_ok": bool(bot._impulse_regime_transition_ok(symbol, "short", profile)),
        }

    group_bot = make_bot(universe(), monkeypatch=monkeypatch)
    del group_bot.compute_group_strengths
    group_bot.industry_map_data = INDUSTRY_ETF_MAP_FILE
    groups = _jsonable(bounce_bot.BounceBot.compute_group_strengths(group_bot))

    return {
        "payloads": payloads,
        "scores": scores,
        "focus_gate": focus,
        "alignment": alignment,
        "impulse": impulse,
        "group_strength": groups,
    }


def _load_fixture():
    if not FIXTURE_PATH.exists():  # pragma: no cover
        pytest.fail(f"golden fixture missing: {FIXTURE_PATH}")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def assert_close(actual, expected, path="$"):
    """Equal, with floats compared to 1e-9 relative (Linux vs Windows libm)."""
    if isinstance(expected, float) or isinstance(actual, float):
        assert isinstance(actual, (int, float)) and isinstance(expected, (int, float)), path
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12), f"{path}: {actual} != {expected}"
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert sorted(actual) == sorted(expected), f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{path}: length"
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            assert_close(left, right, f"{path}[{index}]")
        return
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


@pytest.mark.skipif(
    not os.environ.get("RRS_INTRADAY_WRITE_FIXTURE"),
    reason="one-off recording on the pre-switch code (RRS_INTRADAY_WRITE_FIXTURE=1)",
)
def test_record_the_golden_fixture(monkeypatch):
    import subprocess

    # The recorded source is scripts/ as it stands; name the pinned commit only
    # when scripts/ is byte-identical to it, otherwise HEAD (which the pin rejects).
    same_source = subprocess.call(
        ["git", "diff", "--quiet", PINNED_SOURCE_COMMIT, "--", "scripts"], cwd=str(ROOT_DIR)
    ) == 0
    commit = (
        PINNED_SOURCE_COMMIT
        if same_source
        else subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT_DIR), text=True).strip()
    )
    recorded = record(monkeypatch)
    recorded["generated_from_commit"] = commit
    recorded["universe_version"] = "rrs_intraday_synthetic_v1"
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def test_the_fixture_was_recorded_before_the_switch():
    assert _load_fixture()["generated_from_commit"] == PINNED_SOURCE_COMMIT


def test_desk_engine_reproduces_the_golden(monkeypatch):
    monkeypatch.setenv("TRADINGBOTV3_RRS_ENGINE", "desk")
    fixture = _load_fixture()
    produced = json.loads(json.dumps(record(monkeypatch)))
    for key in ("payloads", "scores", "focus_gate", "alignment", "impulse", "group_strength"):
        assert_close(produced[key], fixture[key], key)


def test_the_golden_universe_is_not_degenerate():
    """The fixture must exercise both sides of the gates it pins."""
    fixture = _load_fixture()
    fifteen = fixture["payloads"]["15m"]
    assert {row[1] for row in fifteen["results"] if row[0] == "RS"} >= {"STRONG"}
    assert {row[1] for row in fifteen["results"] if row[0] == "RW"} == {"WEAK"}
    assert fifteen["results_sector"] and fifteen["results_industry"]
    assert set(fixture["focus_gate"].values()) == {True, False}
    assert any(item["profile"] for item in fixture["impulse"].values())
    assert fixture["group_strength"]["M5"]["sectors"]


# --- the rolling engine on the same universe: facts, not a recording ---------


@pytest.fixture
def rolling_engine(monkeypatch):
    monkeypatch.setenv("TRADINGBOTV3_RRS_ENGINE", "rolling_hourly")


def _scan(monkeypatch, last_day_bars=BARS_PER_SESSION):
    bot = make_bot(universe(last_day_bars), monkeypatch=monkeypatch)
    bot.run_rrs_scan()
    return bot


def _signals(payload, key="results"):
    return {row[1]: row[0] for row in payload[key]}


def test_rolling_scan_names_the_strong_and_weak_stocks(rolling_engine, monkeypatch):
    bot = _scan(monkeypatch)
    for key in CYCLE_TIMEFRAMES:
        payload = bot.rrs_payload_for(key)
        assert payload["threshold"] == 1.0
        assert _signals(payload) == {"STRONG": "RS", "WEAK": "RW"}, key
        reads = {symbol: value[0] for symbol, value in payload["rrs_all"].items()}
        assert reads["STRONG"] > reads["FLAT"] > reads["WEAK"], key
        assert abs(reads["FLAT"]) < 1.0, key


def test_rolling_scan_lets_a_burst_decay(rolling_engine, monkeypatch):
    """BURST ran for 90 minutes then went flat: RS early, not RS at the close."""
    early = _scan(monkeypatch, last_day_bars=20).rrs_payload_for("5m")
    assert early["rrs_all"]["BURST"][0] >= 1.0
    late = _scan(monkeypatch).rrs_payload_for("5m")
    assert abs(late["rrs_all"]["BURST"][0]) < 1.0
    assert "BURST" not in _signals(late)


def test_rolling_first_read_is_no_earlier_than_70_minutes_after_the_open(rolling_engine, monkeypatch):
    session_open = datetime(2026, 6, 5, SESSION_OPEN_HOUR, SESSION_OPEN_MINUTE)
    first = None
    for count in range(1, 30):
        if _scan(monkeypatch, last_day_bars=count).rrs_payload_for("5m")["rrs_all"]:
            first = count
            break
    assert first is not None
    last_bar_open = session_open + timedelta(minutes=5 * (first - 1))
    assert last_bar_open >= session_open + timedelta(minutes=70)


def test_rolling_sector_and_industry_reads_are_rolling_too(rolling_engine, monkeypatch):
    payload = _scan(monkeypatch).rrs_payload_for("5m")
    assert _signals(payload, "results_sector") == {"STRONG": "RS", "WEAK": "RW"}
    assert _signals(payload, "results_industry") == {"STRONG": "RS", "WEAK": "RW"}
    for scope in ("rrs_sector_all", "rrs_industry_all"):
        assert abs(payload[scope]["FLAT"][0]) < 1.0


def test_rolling_score_bonus_rewards_the_aligned_leader(rolling_engine, monkeypatch):
    bot = _scan(monkeypatch)
    bot.latest_rrs_payload = bot.rrs_payload_for("5m")

    def score(symbol, direction):
        context = bot._build_bounce_context_snapshot(symbol, direction)
        assert context["rrs_engine"] == "rolling_hourly"
        return bot._score_bounce_candidate_snapshot(direction, PROBE_LEVELS, context)

    assert score("STRONG", "long") > score("FLAT", "long")
    assert score("WEAK", "short") > score("FLAT", "short")


def test_rolling_impulse_profile_reads_today_with_the_history_behind_it(rolling_engine, monkeypatch):
    bot = _scan(monkeypatch)
    bars = universe()
    last_day = SESSION_DATES[-1]
    today = [bar for bar in bars["STRONG"] if bar.dt.date() == last_day]
    profile = bot._impulse_rrs_profile("STRONG", today, bars[SPY], last_day)
    assert profile
    assert all(item["dt"].date() == last_day for item in profile)
    session_open = datetime(2026, 6, 5, SESSION_OPEN_HOUR, SESSION_OPEN_MINUTE)
    assert profile[0]["dt"] >= session_open + timedelta(minutes=70)
    assert all(item["rrs"] > 0 for item in profile)


def test_rolling_group_strength_tags_its_scale(rolling_engine, monkeypatch):
    bot = make_bot(universe(), monkeypatch=monkeypatch)
    del bot.compute_group_strengths
    groups = bounce_bot.BounceBot.compute_group_strengths(bot)
    for key in ("M5", "H1"):
        assert groups[key]["rrs_engine"] == "rolling_hourly"
        assert groups[key]["sectors"] and groups[key]["industries"]
