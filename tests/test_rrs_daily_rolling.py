"""Facts the rolling RRS must hold at every daily RS site and in rs_engine_v2.

Inputs are the fixed synthetic bars of ``test_rrs_daily_golden``; the checks
are facts (who is RS/RW, ordering, decay, unknown, side symmetry), not a
recording of the new code's own output.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
TESTS_DIR = ROOT_DIR / "tests"
for path in (SCRIPTS_DIR, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rrs_config  # noqa: E402
import test_rrs_daily_golden as golden  # noqa: E402


@pytest.fixture(autouse=True)
def _rolling_engine(monkeypatch):
    monkeypatch.delenv("TRADINGBOTV3_RRS_ENGINE", raising=False)
    monkeypatch.setattr(rrs_config, "RRS_ENGINE", rrs_config.ENGINE_ROLLING)


CUTOFF = rrs_config.ROLLING_RRS_CUTOFF


def _rows(symbol):
    from master_avwap_lib import legacy

    return legacy._daily_frame_to_rows(golden.daily_frame(symbol))


def _last(rows):
    return date.fromisoformat(rows[-1]["date"])


# --- RS Window daily strength ----------------------------------------------------

@pytest.fixture
def strength_map(tmp_path, monkeypatch):
    import setup_playbook_study
    from master_avwap_lib import legacy
    from ui.services import rs_window_feed

    frames = golden.daily_frames()
    monkeypatch.setattr(setup_playbook_study, "_load_daily_frame", lambda symbol: frames.get(symbol))
    monkeypatch.setattr(legacy, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    rs_window_feed._daily_strength_cache.clear()
    yield rs_window_feed.daily_strength_map(list(golden.STOCKS))
    rs_window_feed._daily_strength_cache.clear()


def test_rs_window_daily_strength_names_rs_and_rw_on_the_cutoff(strength_map):
    assert strength_map["STRONG"]["d1_rs_5d"] >= CUTOFF
    assert strength_map["WEAK"]["d1_rs_5d"] <= -CUTOFF
    assert abs(strength_map["FLAT"]["d1_rs_5d"]) < CUTOFF
    assert strength_map["STRONG"]["d1_rs_20d"] > strength_map["FLAT"]["d1_rs_20d"] > strength_map["WEAK"]["d1_rs_20d"]


def test_rs_window_daily_strength_burst_then_flat_has_left_the_5d_read(strength_map):
    # The +10% jump was nine sessions ago: the 20-day read still holds it, the 5-day does not.
    assert strength_map["BURST"]["d1_rs_20d"] >= CUTOFF
    assert abs(strength_map["BURST"]["d1_rs_5d"]) < CUTOFF


def test_rs_window_daily_strength_is_unknown_when_history_is_short(strength_map):
    assert strength_map["NEWCO"]["d1_rs_5d"] is None
    assert strength_map["NEWCO"]["d1_rs_20d"] is None
    assert "weekly_streak" in strength_map["NEWCO"]


def test_rs_window_daily_strength_is_unknown_without_spy(tmp_path, monkeypatch):
    import setup_playbook_study
    from master_avwap_lib import legacy
    from ui.services import rs_window_feed

    frames = {k: v for k, v in golden.daily_frames().items() if k != "SPY"}
    monkeypatch.setattr(setup_playbook_study, "_load_daily_frame", lambda symbol: frames.get(symbol))
    monkeypatch.setattr(legacy, "MASTER_AVWAP_DAILY_BARS_DIR", tmp_path)
    rs_window_feed._daily_strength_cache.clear()
    try:
        values = rs_window_feed.daily_strength_map(["STRONG"])["STRONG"]
    finally:
        rs_window_feed._daily_strength_cache.clear()
    assert values["d1_rs_5d"] is None and values["d1_rs_20d"] is None


# --- the rolling daily read itself -------------------------------------------------

def test_daily_burst_then_flat_decays_session_by_session():
    from daily_rrs import daily_rolling_rrs

    burst, spy = _rows("BURST"), _rows("SPY")
    burst_day = 80  # index of the +10 jump in the synthetic series
    reads = [
        daily_rolling_rrs(burst, spy, rrs_config.DAILY_5D, through=burst[burst_day + k]["date"]).rolling
        for k in range(0, 10)
    ]
    peak = max(range(len(reads)), key=lambda k: reads[k])
    assert reads[peak] >= CUTOFF
    assert all(reads[k] >= reads[k + 1] for k in range(peak, len(reads) - 1)), reads
    assert abs(reads[-1]) < CUTOFF


def test_daily_read_needs_the_symbols_last_day_in_spy():
    from daily_rrs import daily_rolling_rrs

    strong, spy = _rows("STRONG"), _rows("SPY")
    assert daily_rolling_rrs(strong, spy[:-1], rrs_config.DAILY_5D) is None
    assert daily_rolling_rrs(strong[:-1], spy, rrs_config.DAILY_5D) is not None


# --- Master AVWAP D1 RS vs SPY -------------------------------------------------------

def test_d1_rs_bonus_goes_to_long_strength_and_short_weakness():
    from master_avwap_lib import legacy

    spy = _rows("SPY")
    strong, weak, flat = _rows("STRONG"), _rows("WEAK"), _rows("FLAT")

    def assess(rows, side):
        return legacy.assess_daily_relative_strength(rows, _last(rows), side, {}, spy_daily_rows=spy)

    assert assess(strong, "LONG")["daily_relative_strength_bonus"] == legacy.DAILY_RELATIVE_STRENGTH_BONUS
    assert assess(strong, "SHORT")["daily_relative_strength_bonus"] == 0
    assert assess(weak, "SHORT")["daily_relative_strength_bonus"] == legacy.DAILY_RELATIVE_STRENGTH_WEAKNESS_BONUS
    assert assess(weak, "LONG")["daily_relative_strength_bonus"] == 0
    for side in ("LONG", "SHORT"):
        assert assess(flat, side)["daily_relative_strength_bonus"] == 0
    assert "rolling RRS" in assess(strong, "LONG")["daily_relative_strength_note"]
    assert (
        assess(strong, "LONG")["daily_relative_strength_score"]
        > assess(flat, "LONG")["daily_relative_strength_score"]
        > assess(weak, "LONG")["daily_relative_strength_score"]
    )


def test_d1_rs_is_unknown_not_zero_without_spy_rows_or_history():
    from master_avwap_lib import legacy

    strong, newco = _rows("STRONG"), _rows("NEWCO")
    no_spy = legacy.assess_daily_relative_strength(strong, _last(strong), "LONG", {"one_day_return_pct": 0.1, "five_day_return_pct": 0.2})
    assert no_spy["daily_relative_strength_score"] is None
    assert no_spy["daily_relative_strength_bonus"] == 0
    short = legacy.assess_daily_relative_strength(newco, _last(newco), "LONG", {}, spy_daily_rows=_rows("SPY"))
    assert short["daily_relative_strength_score"] is None
    assert short["symbol_five_day_return_pct"] is not None  # the plain returns are still reported


def test_d1_rs_is_point_in_time():
    from master_avwap_lib import legacy

    strong, spy = _rows("STRONG"), _rows("SPY")
    as_of = _last(strong) - timedelta(days=14)
    cut = [row for row in strong if row["date"] <= as_of.isoformat()]
    full = legacy.assess_daily_relative_strength(strong, as_of, "LONG", {}, spy_daily_rows=spy)
    trimmed = legacy.assess_daily_relative_strength(cut, as_of, "LONG", {}, spy_daily_rows=[r for r in spy if r["date"] <= as_of.isoformat()])
    assert full["daily_relative_strength_score"] == trimmed["daily_relative_strength_score"]


def test_d1_rs_mirrors_for_a_mirrored_world():
    from master_avwap_lib import legacy

    def mirror(rows, pivot):
        return [
            {**row, "open": 2 * pivot - row["open"], "high": 2 * pivot - row["low"],
             "low": 2 * pivot - row["high"], "close": 2 * pivot - row["close"]}
            for row in rows
        ]

    strong, spy = _rows("STRONG"), _rows("SPY")
    up = legacy.assess_daily_relative_strength(strong, _last(strong), "LONG", {}, spy_daily_rows=spy)
    down = legacy.assess_daily_relative_strength(
        mirror(strong, 300.0), _last(strong), "SHORT", {}, spy_daily_rows=mirror(spy, 1200.0)
    )
    assert down["daily_relative_strength_score"] == pytest.approx(-up["daily_relative_strength_score"], abs=1e-3)
    assert down["daily_relative_strength_bonus"] == up["daily_relative_strength_bonus"] > 0


# --- Master AVWAP industry RS ----------------------------------------------------------

def _universe(context_etf="XLK", symbol="STRONG"):
    from master_avwap_lib import legacy

    frames = golden.daily_frames()
    return legacy.build_universe_strength_rows(
        {symbol: frames[symbol]},
        {},
        sides_by_symbol={symbol: "LONG"},
        industry_context_by_symbol={symbol: {"industry_etf": context_etf}},
        industry_daily_frames_by_etf={context_etf: frames[context_etf]},
        spy_daily_rows=_rows("SPY"),
    )[0]


def test_stock_vs_industry_is_daily_rrs_against_the_etf():
    strong_vs_falling_etf = _universe("XLE", "STRONG")
    weak_vs_rising_etf = _universe("XLK", "WEAK")
    assert strong_vs_falling_etf["rs_vs_industry"] >= CUTOFF
    assert strong_vs_falling_etf["rs_vs_industry_1d"] > 0 and strong_vs_falling_etf["rs_vs_industry_5d"] > 0
    assert weak_vs_rising_etf["rs_vs_industry"] <= -CUTOFF
    assert _universe("XLK", "NEWCO")["rs_vs_industry"] is None  # 28 days: too short to roll


def test_industry_bonus_thresholds_sit_on_the_rolling_cutoff():
    from master_avwap_lib import legacy

    def bonus(rs_vs_industry, industry_rs, side="LONG"):
        row = {"industry_etf": "XLK", "rs_vs_industry": rs_vs_industry, "rs_vs_industry_1d": -0.1, "rs_vs_industry_5d": 0.1}
        if side == "SHORT":
            row = {**row, "rs_vs_industry_1d": 0.1, "rs_vs_industry_5d": -0.1}
        return legacy.assess_industry_relative_strength(
            row, side, {"industry_etf": "XLK", "daily_relative_strength_score": industry_rs}, scoring_enabled=True
        )["industry_relative_strength_bonus"]

    base = legacy.INDUSTRY_RELATIVE_STRENGTH_BONUS
    assert bonus(CUTOFF, 0.5 * CUTOFF) == base
    assert bonus(CUTOFF - 0.01, 0.5 * CUTOFF) == 0
    assert bonus(CUTOFF, 0.5 * CUTOFF - 0.01) == 0
    assert bonus(-CUTOFF, -0.5 * CUTOFF, "SHORT") == base
    assert bonus(-CUTOFF + 0.01, -0.5 * CUTOFF, "SHORT") == 0


# --- rs_engine_v2 --------------------------------------------------------------------

def _m5(closes, *, pad=0.05, sessions=golden.M5_SESSIONS):
    from market_state import M5Bar

    bars, prev = [], closes[0]
    for k, close in enumerate(closes):
        session, j = divmod(k, golden.M5_BARS_PER_SESSION)
        day = sessions[session]
        ts = datetime(day.year, day.month, day.day, 6, 35) + timedelta(minutes=5 * j)
        bars.append(M5Bar(ts=ts, open=prev, high=max(prev, close) + pad, low=min(prev, close) - pad, close=close, volume=1.0))
        prev = close
    return bars


def _engine_world():
    from relative_strength import CandidateInput

    total = len(golden.M5_SESSIONS) * golden.M5_BARS_PER_SESSION
    last_session = total - golden.M5_BARS_PER_SESSION
    spy = _m5([500.0 + 0.1 * ((k * 7) % 5 - 2) for k in range(total)], pad=0.25)
    def drift(k, per_bar):
        return per_bar * (k - last_session) if k >= last_session else 0.0

    leader = [100.0 + 0.1 * ((k * 3) % 4 - 1.5) + drift(k, 0.05) for k in range(total)]
    laggard = [100.0 + 0.1 * ((k * 3) % 4 - 1.5) - drift(k, 0.05) for k in range(total)]
    flat = [100.0 + 0.1 * ((k * 5) % 4 - 1.5) for k in range(total)]
    candidates = [
        CandidateInput(symbol="LEAD", side_sign=1, stock_bars=_m5(leader)),
        CandidateInput(symbol="FLAT1", side_sign=1, stock_bars=_m5(flat)),
        CandidateInput(symbol="FLAT2", side_sign=1, stock_bars=_m5([c + drift(k, 0.004) for k, c in enumerate(flat)])),
        CandidateInput(symbol="FLAT3", side_sign=1, stock_bars=_m5([c - drift(k, 0.004) for k, c in enumerate(flat)])),
        CandidateInput(symbol="LAG", side_sign=1, stock_bars=_m5(laggard)),
    ]
    return spy, candidates


def test_engine_v2_names_the_leader_defiant_and_the_laggard_fading():
    from relative_strength import RelativeStrengthEngine

    spy, candidates = _engine_world()
    ranks = {r.symbol: r for r in RelativeStrengthEngine().rank(spy, candidates)}
    assert ranks["LEAD"].rolling_rrs >= CUTOFF
    assert ranks["LAG"].rolling_rrs <= -CUTOFF
    assert ranks["LEAD"].tier == "DEFIANT"
    assert ranks["LAG"].tier == "FADING"
    assert all(abs(ranks[s].rolling_rrs) < CUTOFF and ranks[s].tier not in ("DEFIANT", "FADING") for s in ("FLAT1", "FLAT2", "FLAT3"))
    assert all(r.engine_version == "rs_engine_v2" for r in ranks.values())
    assert ranks["LEAD"].components["residual"] > ranks["FLAT1"].components["residual"] > ranks["LAG"].components["residual"]


def test_engine_v2_one_session_is_unknown_and_never_extreme():
    from relative_strength import CandidateInput, RelativeStrengthEngine

    spy, candidates = _engine_world()
    today = golden.M5_BARS_PER_SESSION
    short = [
        CandidateInput(symbol=c.symbol, side_sign=c.side_sign, stock_bars=c.stock_bars[-today:]) for c in candidates
    ]
    ranks = RelativeStrengthEngine().rank(spy[-today:], short)
    assert all(r.rolling_rrs is None for r in ranks)
    assert all(r.tier not in ("DEFIANT", "FADING") for r in ranks)


def test_engine_v2_burst_decays_within_the_session():
    from relative_strength import candidate_rolling_rrs

    total = len(golden.M5_SESSIONS) * golden.M5_BARS_PER_SESSION
    last_session = total - golden.M5_BARS_PER_SESSION
    spy = _m5([500.0 + 0.1 * ((k * 7) % 5 - 2) for k in range(total)], pad=0.25)
    burst = _m5([100.0 + 0.1 * ((k * 3) % 4 - 1.5) + (2.0 if k >= last_session + 20 else 0.0) for k in range(total)])
    soon = candidate_rolling_rrs(burst[: last_session + 30], spy[: last_session + 30])
    later = candidate_rolling_rrs(burst, spy)
    assert soon >= CUTOFF
    assert abs(later) < CUTOFF


def test_engine_v2_long_short_mirror_equivalence():
    from market_state import mirror_bar
    from relative_strength import RelativeStrengthEngine, mirror_candidate

    spy, candidates = _engine_world()
    candidates[1].d1_rrs = 1.3
    candidates[4].d1_rrs = -0.7
    engine = RelativeStrengthEngine()
    long_ranks = engine.rank(spy, candidates)
    short_ranks = engine.rank([mirror_bar(b, 500.0) for b in spy], [mirror_candidate(c, 100.0) for c in candidates])
    assert [r.symbol for r in short_ranks] == [r.symbol for r in long_ranks]
    for long_rank, short_rank in zip(long_ranks, short_ranks, strict=True):
        assert short_rank.tier == long_rank.tier
        assert short_rank.composite == pytest.approx(long_rank.composite, abs=1e-9)
        assert short_rank.side_sign == -long_rank.side_sign


def test_engine_v2_d1_component_reads_the_daily_rrs():
    from relative_strength import RelativeStrengthEngine

    spy, candidates = _engine_world()
    candidates[1].d1_excess_return_pct = -50.0  # the desk input no longer counts
    candidates[1].d1_rrs = 2.0
    candidates[2].d1_rrs = -2.0
    ranks = {r.symbol: r for r in RelativeStrengthEngine().rank(spy, candidates)}
    assert ranks["FLAT1"].components["d1_strength"] > ranks["FLAT2"].components["d1_strength"]


def test_desk_switch_puts_the_engine_back_on_v1(monkeypatch):
    from relative_strength import RelativeStrengthEngine

    monkeypatch.setenv("TRADINGBOTV3_RRS_ENGINE", "desk")
    spy, candidates = _engine_world()
    ranks = RelativeStrengthEngine().rank(spy, candidates)
    assert all(r.engine_version == "rs_engine_v1" and r.rolling_rrs is None for r in ranks)
