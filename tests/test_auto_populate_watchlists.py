"""Tests for the universe -> longs.txt/shorts.txt auto-populate engine."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from watchlist_utils import read_watchlist_symbols  # noqa: E402


def test_auto_populate_caps_follow_regime():
    from autopilot_core import auto_populate_caps

    assert auto_populate_caps("bullish_strong") == (150, 50)
    assert auto_populate_caps("bullish_weak") == (150, 50)
    assert auto_populate_caps("bearish_strong") == (50, 150)
    assert auto_populate_caps("bearish_weak") == (50, 150)
    assert auto_populate_caps("neutral_chop") == (100, 100)
    assert auto_populate_caps("") == (100, 100)


def _ctx(prev_high, prev_low, prev_close, adr):
    return {"prev_high": prev_high, "prev_low": prev_low, "prev_close": prev_close, "adr": adr}


def _profile(last, *, at_high=0.0, at_low=0.0):
    return {
        "last": last,
        "day_high": last,
        "day_low": last,
        "time_at_high_frac": at_high,
        "time_at_low_frac": at_low,
    }


def test_adr_breakout_candidates_require_break_and_adr_move():
    from autopilot_core import build_adr_breakout_candidates

    profiles = {
        # PDH break + 1.5 ADR move + lots of time at HOD -> strong long.
        "AAA": _profile(103.0, at_high=0.8),
        # PDH break but a tiny move vs ADR -> rejected.
        "BBB": _profile(100.6, at_high=0.9),
        # Big move but still under PDH -> rejected.
        "CCC": _profile(99.9, at_high=0.5),
        # PDL break with -1.0 ADR -> short.
        "DDD": _profile(96.0, at_low=0.6),
        "SPY": _profile(500.0, at_high=1.0),
    }
    context = {
        "AAA": _ctx(100.5, 98.0, 100.0, 2.0),
        "BBB": _ctx(100.5, 98.0, 100.0, 2.0),
        "CCC": _ctx(100.5, 98.0, 100.0, 2.0),
        "DDD": _ctx(100.5, 98.0, 100.0, 4.0),
        "SPY": _ctx(499.0, 490.0, 495.0, 5.0),
    }

    result = build_adr_breakout_candidates(profiles, context)
    long_symbols = [row["symbol"] for row in result["longs"]]
    short_symbols = [row["symbol"] for row in result["shorts"]]

    assert long_symbols == ["AAA"]  # SPY is excluded even when qualifying
    assert short_symbols == ["DDD"]
    assert "PDH break" in result["longs"][0]["reason"]
    assert "+1.5 ADR" in result["longs"][0]["reason"]
    assert "80% of day at HOD" in result["longs"][0]["reason"]
    assert "PDL break" in result["shorts"][0]["reason"]


def test_adr_candidates_rank_by_move_plus_extreme_time():
    import pytest

    from autopilot_core import build_adr_breakout_candidates

    profiles = {
        "FAST": _profile(104.0, at_high=0.1),  # 2.0 ADR + 0.1 -> ~2.1
        "GRIND": _profile(102.4, at_high=0.9),  # 1.2 ADR + 0.9 -> ~2.1 (grinder ties the spike)
        "MEH": _profile(101.6, at_high=0.2),  # 0.8 ADR + 0.2 -> 1.0
    }
    context = {sym: _ctx(100.5, 98.0, 100.0, 2.0) for sym in profiles}

    result = build_adr_breakout_candidates(profiles, context)
    scores = {row["symbol"]: row["score"] for row in result["longs"]}
    assert scores["FAST"] == pytest.approx(2.1)
    assert scores["GRIND"] == pytest.approx(2.1)
    assert scores["MEH"] == pytest.approx(1.0)
    assert [row["symbol"] for row in result["longs"]][-1] == "MEH"


def test_apply_auto_populate_rotates_only_owned_names(tmp_path, monkeypatch):
    import autopilot_core as core
    from candidate_registry import CandidateRegistry

    apply_auto_populated_watchlists = core.apply_auto_populated_watchlists
    record_auto_watchlist_cut = core.record_auto_watchlist_cut

    longs_path = tmp_path / "longs.txt"
    shorts_path = tmp_path / "shorts.txt"
    membership_path = tmp_path / "auto_membership.json"
    registry_path = tmp_path / "candidate_registry.json"
    monkeypatch.setattr(core, "candidate_registry_path", lambda: registry_path)
    longs_path.write_text("TRADER1\nTRADER2\n", encoding="utf-8")
    shorts_path.write_text("", encoding="utf-8")

    def candidates(long_syms, short_syms=()):
        return {
            "longs": [{"symbol": s, "score": 10 - i, "reason": "PDH break"} for i, s in enumerate(long_syms)],
            "shorts": [{"symbol": s, "score": 10 - i, "reason": "PDL break"} for i, s in enumerate(short_syms)],
        }

    first = apply_auto_populated_watchlists(
        candidates(["AAA", "BBB", "TRADER1"], ["ZZZ"]),
        "neutral_chop",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=membership_path,
    )
    # Trader names stay first; TRADER1 is not double-added as an auto name.
    assert read_watchlist_symbols(longs_path) == ["TRADER1", "TRADER2", "AAA", "BBB"]
    assert read_watchlist_symbols(shorts_path) == ["ZZZ"]
    assert first["long"]["added"] == ["AAA", "BBB"]
    assert first["long"]["trader_names"] == 2
    registry = CandidateRegistry.load(registry_path)
    assert registry.get("AAA", "LONG").memberships["auto_populate"].lease_expires_at
    assert "auto_populate" in registry.get("ZZZ", "SHORT").memberships
    assert registry.get("TRADER1", "LONG") is None

    # Next refresh: BBB drops out, CCC arrives; trader names untouched.
    second = apply_auto_populated_watchlists(
        candidates(["AAA", "CCC"], ["ZZZ"]),
        "neutral_chop",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=membership_path,
    )
    assert read_watchlist_symbols(longs_path) == ["TRADER1", "TRADER2", "AAA", "CCC"]
    assert second["long"]["rotated_out"] == ["BBB"]
    assert second["long"]["added"] == ["CCC"]
    registry = CandidateRegistry.load(registry_path)
    assert not registry.get("BBB", "LONG").active
    assert "auto_populate" in registry.get("CCC", "LONG").memberships

    # A VWAP-cut name is blacklisted for the day and not re-added.
    record_auto_watchlist_cut("AAA", "long", membership_path=membership_path)
    registry = CandidateRegistry.load(registry_path)
    assert not registry.get("AAA", "LONG").active
    third = apply_auto_populated_watchlists(
        candidates(["AAA", "CCC"], ["ZZZ"]),
        "neutral_chop",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=membership_path,
    )
    assert "AAA" not in read_watchlist_symbols(longs_path)
    assert third["long"]["total_auto"] == 1  # just CCC


def test_apply_auto_populate_respects_regime_caps_and_side_exclusivity(tmp_path):
    from autopilot_core import apply_auto_populated_watchlists

    longs_path = tmp_path / "longs.txt"
    shorts_path = tmp_path / "shorts.txt"
    membership_path = tmp_path / "auto_membership.json"

    long_rows = [{"symbol": f"L{i:03d}", "score": 500 - i, "reason": "PDH"} for i in range(200)]
    short_rows = [{"symbol": f"S{i:03d}", "score": 500 - i, "reason": "PDL"} for i in range(200)]
    # A symbol appearing on both sides only keeps its first (long) placement.
    short_rows.insert(0, {"symbol": "L000", "score": 999, "reason": "PDL"})

    summary = apply_auto_populated_watchlists(
        {"longs": long_rows, "shorts": short_rows},
        "bullish_strong",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=membership_path,
    )
    assert summary["caps"] == (150, 50)
    assert len(read_watchlist_symbols(longs_path)) == 150
    shorts = read_watchlist_symbols(shorts_path)
    assert len(shorts) == 50
    assert "L000" not in shorts

    bearish = apply_auto_populated_watchlists(
        {"longs": long_rows, "shorts": short_rows},
        "bearish_weak",
        longs_path=longs_path,
        shorts_path=shorts_path,
        membership_path=membership_path,
    )
    assert bearish["caps"] == (50, 150)
    assert len(read_watchlist_symbols(longs_path)) == 50
    assert len(read_watchlist_symbols(shorts_path)) == 150
