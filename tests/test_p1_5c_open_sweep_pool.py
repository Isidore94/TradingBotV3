"""P1-5 5c: the open sweep reads all of universe_all.txt; Focus and typed names first.

The IB budget is unchanged: the sweep is yfinance, and only the top
AUTOPILOT_WATCHLIST_CAP names per side by gap and RS reach the bot's files,
beside every name the trader typed (merge_autopilot_watchlist keeps them).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import autopilot_core as core  # noqa: E402


def _write(path: Path, symbols) -> Path:
    path.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    return path


def _universe(tmp_path, monkeypatch, *, longs, shorts, all_names):
    monkeypatch.setattr(core, "UNIVERSE_LONGS_FILE", _write(tmp_path / "ul.txt", longs))
    monkeypatch.setattr(core, "UNIVERSE_SHORTS_FILE", _write(tmp_path / "us.txt", shorts))
    monkeypatch.setattr(core, "UNIVERSE_ALL_FILE", _write(tmp_path / "ua.txt", all_names))


def test_every_universe_all_name_is_swept_when_the_lists_overflow_the_old_cap(tmp_path, monkeypatch):
    """Live 2026-09-24: 1,163 long/short names + 1,467 in universe_all; the old 1,200 cap cut 267."""
    longs = [f"L{i:04d}" for i in range(500)]
    shorts = [f"S{i:04d}" for i in range(700)]
    extra = [f"A{i:04d}" for i in range(300)]
    _universe(tmp_path, monkeypatch, longs=longs, shorts=shorts, all_names=longs + shorts + extra)
    pool = core.load_universe_pool()
    assert set(extra) <= set(pool)
    assert len(pool) == 1500 and len(set(pool)) == 1500


def test_focus_and_typed_names_lead_and_are_never_cut(tmp_path, monkeypatch):
    all_names = [f"A{i:04d}" for i in range(50)]
    _universe(tmp_path, monkeypatch, longs=[], shorts=[], all_names=all_names)
    pool = core.load_universe_pool(max_symbols=10, priority=["myfocus", "TYPED", "A0003", "TYPED"])
    assert pool[:3] == ["MYFOCUS", "TYPED", "A0003"]
    assert len(pool) == 10
    # Even a cap smaller than the priority list keeps every priority name.
    tight = core.load_universe_pool(max_symbols=1, priority=["ONE", "TWO"])
    assert tight == ["ONE", "TWO"]


def test_universe_all_order_leads_the_pool_after_the_priority_names(tmp_path, monkeypatch):
    _universe(tmp_path, monkeypatch, longs=["ZLONG"], shorts=["ZSHORT"], all_names=["AAA", "BBB"])
    assert core.load_universe_pool() == ["AAA", "BBB", "ZLONG", "ZSHORT"]


def test_the_ib_budget_cap_is_unchanged():
    assert core.AUTOPILOT_WATCHLIST_CAP == 40


def test_the_service_pools_focus_names_but_never_typed_names(monkeypatch):
    from ui.services import autopilot_service as svc_mod
    from ui.services.autopilot_service import AutopilotService

    service = AutopilotService.__new__(AutopilotService)
    service._read_watchlists = lambda: (["TYPEDL"], ["TYPEDS"])
    monkeypatch.setattr(
        svc_mod, "read_watchlist_symbols", lambda path: ["FOCUSL"] if "long" in str(path).lower() else ["FOCUSS"]
    )
    names = service._open_sweep_priority_names()
    assert names == ["FOCUSL", "FOCUSS"]


# --- a typed name is the trader's: never an auto pick, never dropped ---------

SPY = {"early_move_pct": 0.0, "gap_pct": 0.0}


def _moves(**gaps):
    moves = {"SPY": SPY}
    for symbol, gap in gaps.items():
        moves[symbol] = {"early_move_pct": gap, "gap_pct": gap}
    return moves


def test_a_typed_name_survives_a_rebuild_after_it_gapped():
    """Reviewer repro: typed ABC gaps +4%, then a flat rebuild must keep ABC."""
    typed_longs = ["ABC"]
    first = core.build_watchlists_from_moves(_moves(ABC=4.0, XYZ=3.0), SPY)
    plan = core.plan_watchlist_write(first["longs"], first["shorts"], typed_longs, [], {})
    assert "ABC" not in plan["written"]["longs"]
    assert plan["merged_longs"]["symbols"] == ["XYZ", "ABC"]
    second = core.build_watchlists_from_moves(_moves(ABC=0.0, XYZ=0.0), SPY)
    plan2 = core.plan_watchlist_write(
        second["longs"], second["shorts"], plan["merged_longs"]["symbols"], [], plan["written"]
    )
    assert plan2["merged_longs"]["symbols"] == ["ABC"]


def test_a_typed_long_that_gaps_down_never_becomes_an_auto_short():
    built = core.build_watchlists_from_moves(_moves(ABC=-4.0, DEF=-3.0), SPY)
    assert "ABC" in built["shorts"]  # the raw ranking would take it
    plan = core.plan_watchlist_write(built["longs"], built["shorts"], ["ABC"], [], {})
    assert plan["shorts"] == ["DEF"]
    assert plan["merged_shorts"]["symbols"] == ["DEF"]
    assert plan["merged_longs"]["symbols"] == ["ABC"]
    assert plan["typed_skipped"] == ["ABC"]


def test_a_typed_name_in_the_universe_is_never_recorded_as_an_auto_pick():
    plan = core.plan_watchlist_write(["UNIV", "AUTO"], ["TYPS"], ["UNIV"], ["TYPS"], {"longs": ["OLD"]})
    assert plan["written"] == {"longs": ["AUTO"], "shorts": []}
    # A name Auto Pilot wrote last time is still its own and may be replaced.
    replaced = core.plan_watchlist_write(["NEW"], [], ["OLD", "MINE"], [], {"longs": ["OLD"]})
    assert replaced["merged_longs"]["symbols"] == ["NEW", "MINE"]


def test_the_service_build_keeps_a_typed_name_across_two_builds(monkeypatch):
    """The whole service path, two builds: typed ABC is never written as auto."""
    from ui.services import autopilot_service as svc_mod
    from ui.services.autopilot_service import AutopilotService

    files = {"longs": ["ABC"], "shorts": []}
    service = AutopilotService.__new__(AutopilotService)
    service._state = {}
    service._building_watchlists = False
    service._log = lambda *_a, **_k: None
    service._save_state = lambda: None
    service._append_pick_rows = lambda *_a, **_k: None
    service._write_report = lambda: None
    service._read_watchlists = lambda: (list(files["longs"]), list(files["shorts"]))
    service._open_sweep_priority_names = lambda: []

    def write(longs, shorts):
        files["longs"], files["shorts"] = list(longs), list(shorts)
        return True

    class _Inline:
        def __init__(self, target=None, **_kwargs):
            self._target = target

        def start(self):
            self._target()

    today = svc_mod.datetime.now().date()
    monkeypatch.setattr(svc_mod.threading, "Thread", _Inline)
    monkeypatch.setattr(core, "load_universe_pool", lambda *a, **k: ["ABC", "XYZ"])
    monkeypatch.setattr(core, "load_daily_context", lambda *_a, **_k: {})
    monkeypatch.setattr(core, "write_bouncebot_watchlists", write)
    monkeypatch.setattr(core, "write_auto_watchlists", lambda *_a, **_k: True)
    gaps = {"ABC": 4.0, "XYZ": 3.0}

    def moves(pool, log=None):
        out = {"SPY": {**SPY, "session_date": today}}
        for symbol in pool:
            out[symbol] = {"early_move_pct": gaps[symbol], "gap_pct": gaps[symbol], "session_date": today}
        return out

    monkeypatch.setattr(core, "fetch_open_scan_moves", moves)
    service._start_watchlist_build(manual=True)
    assert "ABC" not in service._state["autopilot_written"]["longs"]
    assert files["longs"] == ["XYZ", "ABC"]
    gaps.update(ABC=0.0, XYZ=-3.0)
    service._start_watchlist_build(manual=True)
    assert "ABC" in files["longs"]
    assert "ABC" not in files["shorts"]
