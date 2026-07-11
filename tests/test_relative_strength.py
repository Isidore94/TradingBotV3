"""Packet C (plan.md sec 16.2/16.6-16.8): side-symmetric relative-strength engine."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from market_state import M5Bar  # noqa: E402
from relative_strength import (  # noqa: E402
    CandidateInput,
    RelativeStrengthEngine,
    compute_window_features,
    mirror_candidate,
)

START = datetime(2026, 7, 10, 10, 0)


def bars(closes, *, skip=(), start=START):
    out = []
    for i, close in enumerate(closes):
        if i in skip:
            continue
        out.append(
            M5Bar(
                ts=start + timedelta(minutes=5 * i),
                open=close,
                high=close + 0.05,
                low=close - 0.05,
                close=close,
                volume=1_000_000,
            )
        )
    return out


def ramp(start_price, end_price, n=13):
    return [start_price + (end_price - start_price) * i / (n - 1) for i in range(n)]


SPY_PULLBACK = ramp(500.0, 497.0)  # SPY gives back 0.6% over the hour


def test_flat_stock_during_spy_pullback_earns_positive_defiance():
    engine = RelativeStrengthEngine()
    candidates = [
        CandidateInput(symbol="STRONG", side_sign=1, stock_bars=bars(ramp(100.0, 100.4))),
        CandidateInput(symbol="FLAT", side_sign=1, stock_bars=bars([100.0] * 13)),
        CandidateInput(symbol="WEAK", side_sign=1, stock_bars=bars(ramp(100.0, 99.2))),
    ]
    ranks = engine.rank(bars(SPY_PULLBACK), candidates)

    order = [r.symbol for r in ranks]
    assert order == ["STRONG", "FLAT", "WEAK"]

    by_symbol = {r.symbol: r for r in ranks}
    flat = by_symbol["FLAT"]
    assert flat.windows[30].aligned_residual_pct > 0, "flat stock vs falling SPY is defiance"
    assert by_symbol["STRONG"].tier == "DEFIANT"
    assert by_symbol["WEAK"].tier == "FADING"


def test_alignment_never_credits_a_move_the_stock_has_no_bars_for():
    # SPY dumps in its final three bars; the stock's feed stops before the
    # dump. Aligned windows end at the last COMMON timestamp, so the stock
    # earns no phantom defiance from bars it does not have.
    spy_closes = [500.0] * 10 + [498.0, 496.5, 495.0]
    stock = CandidateInput(
        symbol="LAGGED",
        side_sign=1,
        stock_bars=bars([100.0] * 10),
    )
    features = compute_window_features(
        stock.stock_bars,
        bars(spy_closes),
        window_minutes=30,
        side_sign=1,
    )
    assert features.last_ts == START + timedelta(minutes=45)  # last common bar
    assert abs(features.aligned_residual_pct) < 1e-9


def test_low_coverage_marks_candidate_watching_with_stale_penalty():
    engine = RelativeStrengthEngine()
    sparse = CandidateInput(
        symbol="SPARSE",
        side_sign=1,
        stock_bars=bars([100.0] * 13, skip=set(range(2, 11))),  # only 4 bars survive
    )
    full = CandidateInput(symbol="FULL", side_sign=1, stock_bars=bars(ramp(100.0, 100.5)))
    ranks = {r.symbol: r for r in engine.rank(bars(SPY_PULLBACK), [sparse, full])}
    assert ranks["SPARSE"].data_ok is False
    assert ranks["SPARSE"].tier == "WATCHING"
    assert "stale_data" in ranks["SPARSE"].penalties
    assert ranks["FULL"].data_ok is True


def test_long_short_mirror_equivalence():
    engine = RelativeStrengthEngine()
    spy_bars = bars(SPY_PULLBACK)
    candidates = [
        CandidateInput(symbol="A", side_sign=1, stock_bars=bars(ramp(100.0, 100.6))),
        CandidateInput(symbol="B", side_sign=1, stock_bars=bars([100.0] * 13)),
        CandidateInput(symbol="C", side_sign=1, stock_bars=bars(ramp(100.0, 99.0))),
    ]
    long_ranks = engine.rank(spy_bars, candidates)

    from market_state import mirror_bar

    mirrored_spy = [mirror_bar(b, 500.0) for b in spy_bars]
    mirrored = [mirror_candidate(c, 100.0) for c in candidates]
    short_ranks = engine.rank(mirrored_spy, mirrored)

    assert [r.symbol for r in short_ranks] == [r.symbol for r in long_ranks]
    for l, s in zip(long_ranks, short_ranks):
        assert s.tier == l.tier
        assert abs(s.composite - l.composite) < 1e-9
        assert s.side_sign == -l.side_sign


def test_full_universe_candidate_can_outrank_watchlist_name():
    engine = RelativeStrengthEngine()
    watchlist_name = CandidateInput(
        symbol="WATCHED", side_sign=1, stock_bars=bars(ramp(100.0, 99.6)), setup_quality=0.9
    )
    universe_name = CandidateInput(
        symbol="FRESH", side_sign=1, stock_bars=bars(ramp(50.0, 50.6)), setup_quality=0.1
    )
    ranks = engine.rank(bars(SPY_PULLBACK), [watchlist_name, universe_name])
    assert ranks[0].symbol == "FRESH", "pure defiance must be able to beat pedigree"


def test_volatility_normalization_prefers_quiet_tape_for_equal_excess():
    # Same raw excess move; the low-volatility name is the stronger signal.
    quiet = CandidateInput(symbol="QUIET", side_sign=1, stock_bars=bars(ramp(100.0, 100.5)))
    wild_closes = []
    for i, base in enumerate(ramp(100.0, 100.5)):
        wild_closes.append(base + (0.6 if i % 2 else -0.6))
    wild = CandidateInput(symbol="WILD", side_sign=1, stock_bars=bars(wild_closes))
    engine = RelativeStrengthEngine()
    ranks = {r.symbol: r for r in engine.rank(bars(SPY_PULLBACK), [quiet, wild])}
    assert (
        ranks["QUIET"].components["residual"] > ranks["WILD"].components["residual"]
    ), "volatility-adjusted residual must beat raw percent excess"


def test_extension_and_earnings_penalties_are_named_components():
    engine = RelativeStrengthEngine()
    extended = CandidateInput(
        symbol="EXT",
        side_sign=1,
        stock_bars=bars(ramp(100.0, 100.8)),
        extension_atr=3.0,
        earnings_within_days=1,
    )
    normal = CandidateInput(symbol="NORM", side_sign=1, stock_bars=bars(ramp(100.0, 100.4)))
    ranks = {r.symbol: r for r in engine.rank(bars(SPY_PULLBACK), [extended, normal])}
    ext = ranks["EXT"]
    assert "extension" in ext.penalties
    assert "imminent_earnings" in ext.penalties
    assert ext.tier in ("WATCHING",), "over-extended defiance is watch-only, not entry evidence"
    assert set(ext.components) == {
        "residual",
        "giveback",
        "persistence",
        "sector_residual",
        "volume_quality",
        "setup_quality",
        "trigger",
    }


def test_structure_failure_is_invalid_regardless_of_strength():
    engine = RelativeStrengthEngine()
    broken = CandidateInput(
        symbol="BROKEN",
        side_sign=1,
        stock_bars=bars(ramp(100.0, 101.0)),
        structure_ok=False,
    )
    ranks = engine.rank(bars(SPY_PULLBACK), [broken])
    assert ranks[0].tier == "INVALID"
